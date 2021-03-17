import streamlit as st
import nltk
import pandas as pd

nltk.download('vader_lexicon')
nltk.download('punkt')

st.set_page_config(page_title='AM', page_icon=None, layout='centered', initial_sidebar_state='auto')

# To hide hamburger (top right corner) and “Made with Streamlit” footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def sentiment_rating_match(df_original):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    df = df_original.copy()
    df.dropna(inplace=True, subset=['Text'])

    df['Text'] = df['Text'].str.replace(r'\d+', '', regex=True)
    df['Text'] = df['Text'].str.replace(r'[^\w\s]+', '', regex=True)
    df['Text'] = df['Text'].str.replace(r'\^[a-zA-Z]\s+', '', regex=True)
    df['Text'] = df['Text'].str.lower()

    df['sentiment'] = df['Text'].apply(lambda x: sia.polarity_scores(x))

    def convert(x):
        if x > 0.5:
            return "positive"
        else:
            return "negative"

    df['result'] = df['sentiment'].apply(lambda x: convert(x['pos']))

    return df_original.iloc[df.loc[(df['result'] == 'positive') & (df['Star'] == 1), :].index, :]


def main():
    st.title("Play store review identifier")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if st.button("Process"):
        if data_file is not None:
            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)

            df = pd.read_csv(data_file)
            st.markdown("#### Positive reviews with 1 star rating")
            st.dataframe(sentiment_rating_match(df))


if __name__ == '__main__':
    main()
