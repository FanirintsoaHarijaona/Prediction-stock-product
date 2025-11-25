import streamlit as st
import pandas as pd
import numpy as np
import pickle

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pandas as pd


def predict(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    raw_df: the data is expected to be in the same format of the data that we used for the training (id_product, quantity ordered, product, location,date)
    Returns a dataframe with predicted quantity per product per zone.
    """
    FEATURES = ['lag_1', 'lag_2', 'rolling_3', 'product_enc', 'zone_enc']
    # Deserialize model and encoders
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("le_product.pkl", "rb") as f:
        le_product = pickle.load(f)
    with open("le_zone.pkl", "rb") as f:
        le_zone = pickle.load(f)

    df = raw_df.copy()
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['year'] = df['Order Date'].dt.year
    df['month'] = df['Order Date'].dt.strftime('%b')   
    df['month_num'] = df['Order Date'].dt.month
    df["city"] = df["Purchase Address"].str.split(",").str[1]
    df["state"] = "("+df["Purchase Address"].str.split(",").str[2].str[:3]+")"
    df["zone_commande"]= df["city"] + df["state"]
    df['product_enc'] = le_product.transform(df['Product'])
    df['zone_enc'] = le_zone.transform(df['zone_commande'])

    # Sort & feature engineering
    df = df.sort_values(['Product','zone_commande','year','month_num'])
    df['lag_1'] = df.groupby(['Product','zone_commande'])['Quantity Ordered'].shift(1)
    df['lag_2'] = df.groupby(['Product','zone_commande'])['Quantity Ordered'].shift(2)
    df['rolling_3'] = df.groupby(['Product','zone_commande'])['Quantity Ordered']\
                        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    df = df.dropna().reset_index(drop=True)

    df['pred'] = model.predict(df[FEATURES]).astype(int)

    pred_per_prod_zone = df.groupby(['Product','zone_commande'])['pred'].sum().reset_index()
    pred_per_prod_zone["pred"] = pred_per_prod_zone["pred"].astype(int)
    pred_per_prod_zone = pred_per_prod_zone.sort_values(['Product','pred'], ascending=[True, False])
    table_text = pred_per_prod_zone.to_string(index=False)
    prompt_text = f"""
    Vous êtes un assistant de gestion de stock. Voici la quantité prévue pour chaque produit et zone:

    {table_text}

    Répondez aux questions suivantes en français:
    1) Pour chaque produit, quelle zone doit recevoir le plus de stock en priorité?
    2) Comparez la quantité prévue entre Dallas et Houston pour chaque produit.
    """
    model_name = "google/flan-t5-small" 

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.0
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = PromptTemplate(
        input_variables=[],
        template=prompt_text
    )

    chain = LLMChain(prompt=prompt, llm=llm)

    response = chain.run({}) 
    return response

if __name__ == "__main__":
    st.set_page_config(page_title="Forecast Quantity per Zone", layout="wide")
    st.title("Product Quantity Forecast per Zone")

    # Upload CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Raw data:")
        st.dataframe(df.head(10))
        predictions = predict(df)
        st.write("Predictions:")
        st.dataframe(predictions)
        product_select = st.selectbox(
            "Select a product to see prediction per zone",
            predictions['Product'].unique()
        )
        filtered = predictions[predictions['Product'] == product_select]
        st.bar_chart(filtered.set_index('zone_commande')['pred_quantity'])
