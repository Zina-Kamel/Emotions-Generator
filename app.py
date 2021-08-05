import streamlit as st
from predict import *
from data import *
import altair as alt
import streamlit.components.v1 as components
import random
from PyDictionary import PyDictionary
from better_profanity import profanity

def main():
    st.title("Emotion and Emoji Detector Web Application")
    st.subheader("What type of NLP service would you like to use?")

    #Textbox for text user is entering
    st.subheader("Enter the text you'd like to analyze.")
    text = st.text_input('Enter text') #text is stored in this variable

    option = st.selectbox('NLP Service',('None', 'Emotion Detection', 'Harmful Words Detector')) #option is stored in this variable

    #Display results of the NLP task
    st.header("Results")
    
    if option == 'Emotion Detection':
        probs, emotion = predict(text)
        proba_df = pd.DataFrame(probs,columns=["sadness","love","anger","surprise","joy","fear"])
        st.header(emojis[emotion])
        st.write('Your emotion is {}'.format(emotion))
        songs_n = random.randint(0,len(songs[emotion])-1)
        sayings_n = random.randint(0,len(sayings[emotion])-1)
        with st.beta_expander('What is your song?'):
            st.write("Your song is:")
            components.iframe(
                songs[emotion][songs_n]
            )
        with st.beta_expander('What can you read?'):
            st.write("Your article is {}".format(sayings[emotion][sayings_n]))
        with st.beta_expander('Probability:'):
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions","probability"]
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
            st.altair_chart(fig, use_container_width=True)

    if option == 'Harmful Words Detector':
        dictionary = PyDictionary()
        censored = profanity.censor(text)
        oArr = text.split(" ")
        cArr = censored.split(" ")
        censoredList = []

        for i in range(0, len(cArr)):
            if '*' in cArr[i]:
                if oArr[i] not in censoredList:
                    censoredList.append(oArr[i])

        if len(censoredList) == 0:
            st.write("no foul language detected.") 

        else:
            wordDict = {}
            for c in censoredList:
                word = dictionary.meaning(c)
                wordDict[c] = word
            summary = ''

            for word in wordDict:
                w = str(wordDict[word])
                temp = word + " " + w.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
                summary = summary + '\n' + temp + '\n'

            st.header("The harmful words have been censored:")
            st.write(censored)
            st.header("Here are the harmful detected words:")
            st.write(summary)

if __name__ == '__main__':
	main()