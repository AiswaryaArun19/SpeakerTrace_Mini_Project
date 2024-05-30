import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import StandardScaler

# Load your Keras model
model = load_model('model.h5')

# Function to extract MFCC features from audio file
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs)
    mfccs_scaled = mfccs_scaled.T.reshape(1, mfccs_scaled.shape[1], -1)
    return mfccs_scaled

# Define speakers, their biographies, and image URLs
biographies = {
    "Benjamin Netanyau": {
        "biography": "Benjamin Netanyau served as the Prime Minister of Israel, navigating complex challenges during his tenure. Born on October 21, 1949, in Tel Aviv, Netanyau's early life was marked by his family's strong Zionist convictions. His father, Benzion Netanyau, was a prominent historian, and his older brother, Yonatan, was a celebrated hero of the Israeli Defense Forces who tragically died in the Entebbe raid.\nNetanyau's educational journey took him to the United States, where he earned degrees in architecture and business from the Massachusetts Institute of Technology (MIT). His time in the U.S. not only broadened his academic horizons but also connected him with influential figures in American politics and finance. These connections would later prove invaluable in his political career.\nReturning to Israel, Netanyau embarked on a political path that saw him rise rapidly within the ranks of the Likud party. Known for his articulate and charismatic oratory, he gained a reputation as a staunch defender of Israeli security and an advocate for economic liberalization. His tenure as Prime Minister was characterized by a firm stance on national security issues, including the controversial expansion of settlements in the West Bank.\nNetanyau's political strategy often involved leveraging his close relationship with the United States. His efforts to maintain strong U.S.-Israel relations were highlighted by numerous high-profile visits to Washington, D.C., and his outspoken opposition to the Iran nuclear deal negotiated during the Obama administration. Domestically, he implemented significant economic reforms that transformed Israel into a high-tech powerhouse, earning it the moniker Startup Nation.\nDespite his successes, Netanyau's leadership was not without controversy. He faced numerous corruption investigations, leading to indictments on charges of bribery, fraud, and breach of trust. These legal battles, coupled with the polarizing nature of his policies, sparked widespread debate and division within Israeli society. Nonetheless, his legacy as one of Israel's longest-serving and most influential leaders remains firmly entrenched in the nation's history.",
        "image_urls": ["https://ilarge.lisimg.com/image/14922039/740full-benjamin-netanyahu.jpg"]
    },
    "Jens Stoltenberg": {
        "biography": "Jens Stoltenberg is the Secretary General of NATO and has emphasized international cooperation. Born on March 16, 1959, in Oslo, Norway, Stoltenberg grew up in a politically active family. His father, Thorvald Stoltenberg, was a prominent diplomat and politician, and his mother, Karin Stoltenberg, was a renowned geneticist. This environment fostered an early interest in politics and public service.\nStoltenberg's political career began in earnest when he joined the Labour Party in his youth. He quickly ascended the party ranks, becoming the leader of the Workers' Youth League. His early political activities were marked by advocacy for social justice, environmental protection, and peace, reflecting his deeply held progressive values.\nIn 2000, Stoltenberg became the Prime Minister of Norway, a position he held twice, first from 2000 to 2001 and then from 2005 to 2013. His leadership was characterized by a commitment to modernizing the Norwegian economy, particularly through investments in technology and renewable energy. Stoltenberg's government also focused on expanding the welfare state, ensuring robust social safety nets for all citizens.\nStoltenberg's tenure as Prime Minister saw Norway navigating complex global issues, including the financial crisis of 2008. His adept handling of the crisis helped stabilize Norway's economy, earning him praise both domestically and internationally. Additionally, he was a vocal advocate for global disarmament and peace, reflecting his belief in diplomacy and international cooperation.\nIn 2014, Stoltenberg was appointed Secretary General of NATO, marking a significant shift in his career. In this role, he has emphasized the importance of collective defense and transatlantic unity. Stoltenberg has also been instrumental in adapting NATO to address modern security challenges, such as cyber threats and the rise of non-state actors. His tenure has strengthened NATO's presence and capabilities, ensuring its continued relevance in a rapidly changing world.",
        "image_urls": ["https://cdn.britannica.com/65/136665-050-E7B53DEF/Jens-Stoltenberg-2009.jpg"]
    },
    "Julia Gillard": {
        "biography": "Julia Gillard served as the Prime Minister of Australia, focusing on education reform and economic management. Born on September 29, 1961, in Barry, Wales, Gillard emigrated to Australia with her family in 1966. Growing up in Adelaide, she demonstrated early academic promise and a keen interest in public service. Her commitment to education and social justice was evident from a young age.\nGillard's political career began in earnest when she joined the Australian Labor Party during her university years. She quickly became active in student politics, serving as President of the Australian Union of Students in 1983. This early experience in leadership and advocacy laid the groundwork for her future political endeavors.\nIn 1998, Gillard was elected to the Australian House of Representatives as the Member for Lalor, representing the western suburbs of Melbourne. She rapidly established herself as a formidable parliamentarian, known for her sharp intellect and strong debating skills. Gillard held various shadow portfolios before becoming the Deputy Leader of the Opposition in 2006.\nIn 2010, Gillard made history by becoming Australia's first female Prime Minister, following a leadership challenge within the Labor Party. Her tenure was marked by significant policy achievements, particularly in the areas of education and health. One of her signature initiatives was the implementation of the National Broadband Network, aimed at transforming Australia's digital infrastructure.\nGillard's government also introduced major reforms in education, including the Gonski funding model, which aimed to provide equitable funding to schools based on student needs. Additionally, she played a pivotal role in the introduction of the National Disability Insurance Scheme (NDIS), which sought to provide comprehensive support for Australians with disabilities.\nDespite her policy successes, Gillard's time as Prime Minister was also characterized by intense political challenges and a highly polarized media environment. She faced constant scrutiny and often personal attacks, particularly around issues of gender. Nonetheless, she remained steadfast in her commitment to her policy agenda and governance.\nIn 2013, Gillard was succeeded by Kevin Rudd following another leadership contest within the Labor Party. After leaving politics, she continued to contribute to public life through various roles, including as Chair of the Global Partnership for Education, advocating for improved access to education worldwide. Gillard's legacy as a trailblazer for women in Australian politics and her contributions to social policy continue to be recognized and celebrated.",
        "image_urls": ["https://pathwaystopolitics.org.au/wp-content/uploads/2022/02/Julia-Gillard.jpg"]
    },
    "Magaret Thatcher": {
        "biography": "Margaret Thatcher, known as the Iron Lady, served as the Prime Minister of the United Kingdom from 1979 to 1990. Born on October 13, 1925, in Grantham, England, Thatcher grew up in a modest household where her father ran a grocery store and was active in local politics. This upbringing instilled in her a strong work ethic and a keen interest in public service.\nThatcher's political career began in earnest after she graduated from Oxford University with a degree in chemistry and later trained as a barrister. She joined the Conservative Party and was elected as the Member of Parliament for Finchley in 1959. Thatcher quickly rose through the ranks, earning a reputation for her determination and clear ideological stance.\nIn 1975, Thatcher became the leader of the Conservative Party, making history as the first woman to lead a major political party in the UK. Her leadership style was characterized by a firm commitment to free-market principles, reduced government intervention, and individual responsibility. These beliefs laid the foundation for what would later be known as Thatcherism.\nIn 1979, Thatcher was elected Prime Minister, ushering in a period of significant economic and social transformation in the UK. Her government implemented a series of radical policies aimed at curbing inflation, reducing the power of trade unions, and privatizing state-owned industries. These measures were controversial and often met with strong opposition, but they fundamentally reshaped the British economy and society.\nThatcher's tenure saw major events, including the Falklands War in 1982, where her decisive leadership bolstered her popularity and reinforced her image as a strong and unyielding leader. Domestically, her policies led to economic growth and a rise in home ownership, but also increased unemployment and social unrest in certain sectors.\nDespite the divisive nature of her policies, Thatcher's impact on the UK was profound. She was a staunch advocate for the principles of free enterprise and deregulation, which influenced not only British politics but also had a global impact. Her close relationship with US President Ronald Reagan helped to shape the international political landscape during the Cold War, promoting a strong alliance between the UK and the US.\nIn 1990, Thatcher's leadership came to an end when she was challenged by members of her own party, leading to her resignation. After leaving office, she remained active in public life, writing her memoirs and giving speeches around the world. Thatcher's legacy is a subject of intense debate; she is praised for revitalizing the British economy and criticized for the social divisions her policies exacerbated.\nMargaret Thatcher passed away on April 8, 2013, but her influence on politics and her iconic status as the Iron Lady endure. Her life and career continue to be studied and discussed, reflecting her significant role in shaping modern Britain.",
        "image_urls": ["http://assets.vogue.com/photos/5891f0bf153ededd21da4d60/master/pass/img-holdingmargaretthatcher_174159682557.jpg"]
    },
    "Nelson Mandela": {
        "biography": "Nelson Mandela was a symbol of peace and reconciliation and a key figure in dismantling apartheid in South Africa. Born on July 18, 1918, in the village of Mvezo in Umtata, then part of South Africa's Cape Province, Mandela grew up in a rural setting. His early life was influenced by the traditions and taboos of his Xhosa heritage and the stories of his ancestors' resistance to colonial rule.\nMandela's formal education began at a local mission school where he first learned English. He later attended the prestigious Healdtown Methodist Boarding School and then the University of Fort Hare, the only Western-style higher education institute for black people in Southern Africa at that time. However, Mandela did not complete his degree there due to his involvement in a student protest. He later completed his law degree via correspondence from the University of South Africa.\nIn 1944, Mandela joined the African National Congress (ANC), a pivotal step in his political journey. He co-founded the ANC Youth League, advocating for a more radical approach to ending apartheid. Mandela's activism soon made him a target for the authorities. In 1961, he helped establish Umkhonto we Sizwe (Spear of the Nation), the armed wing of the ANC, which aimed to overthrow the apartheid regime through sabotage and guerrilla warfare.\nMandela was arrested in 1962 and, in 1964, was sentenced to life imprisonment during the Rivonia Trial. He spent 27 years in prison, primarily on Robben Island. Despite the harsh conditions, Mandela's resolve remained unbroken. He became a symbol of resistance and resilience, garnering international support for his release and the anti-apartheid cause.\nIn 1990, under increasing domestic and international pressure, South African President F.W. de Klerk released Mandela from prison. Mandela's release marked the beginning of a new era for South Africa. He led the ANC in negotiations with the apartheid government, culminating in the country's first multiracial elections in 1994.\nMandela was elected South Africa's first black president in 1994. His presidency was marked by efforts to heal the deep racial divides and to foster national reconciliation. He established the Truth and Reconciliation Commission, chaired by Archbishop Desmond Tutu, to address the atrocities committed during apartheid. Mandela's leadership emphasized forgiveness, understanding, and a commitment to building a united South Africa.\nMandela's influence extended far beyond South Africa. He became a global icon for peace and justice, receiving numerous awards and honors, including the Nobel Peace Prize in 1993. His autobiography, Long Walk to Freedom, provides a detailed account of his life and the struggle against apartheid.\nAfter stepping down as president in 1999, Mandela continued to work through the Nelson Mandela Foundation to address issues such as HIV/AIDS, rural development, and education. He remained active in advocating for human rights and social justice until his retirement from public life in 2004.\nNelson Mandela passed away on December 5, 2013, at the age of 95. His legacy lives on, celebrated annually on Nelson Mandela International Day, which encourages people to dedicate 67 minutes of their time to public service, representing the 67 years Mandela spent fighting for justice. His life and work continue to inspire millions worldwide, embodying the values of resilience, forgiveness, and the unyielding pursuit of equality and human dignity.",
        "image_urls": ["https://wallpaperbat.com/img/1950119-nelson-mandela.jpg"]
    }
}

def main():
    st.sidebar.title("Instructions")
    st.sidebar.header("How to Use This App")
    st.sidebar.write("""
        - **Step 1**: Upload an audio file (WAV format).
        - **Step 2**: Press the 'Identify Speaker' button.
        - **Step 3**: View the identification results.
    """)
    with st.sidebar:
        if st.button('Learn More'):
            st.write("""
                **What Happens Behind the Scenes:**
                1. **Voice Activity Detection**: The system first checks if the uploaded audio contains human speech.
                2. **Feature Extraction**: It then extracts audio features specifically, Mel-Frequency Cepstral Coefficients (MFCCs), which are crucial for identifying unique voice patterns.
                3. **Model Prediction**: These features are fed into a deep learning model which predicts the most likely speaker based on the trained data.
                4. **Result Display**: Finally, the identified speaker's name and bio are displayed.
            """)

    st.title("SpeakerTrace")

    if 'page' not in st.session_state:
        st.session_state.page = 'upload'

    if st.session_state.page == 'upload':
        uploaded_file = st.file_uploader("Choose an audio file...", type="wav")
        if st.button('Identify Speaker'):
            if uploaded_file is not None:
                temp_dir = 'tempDir'
                os.makedirs(temp_dir, exist_ok=True)
                audio_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                features = extract_features(audio_path)
                predictions = model.predict(features)
                predicted_speaker_index = np.argmax(predictions)
                max_probability = np.max(predictions)

                if max_probability < 0.97:
                    st.error("Speaker not identified")
                else:
                    speaker_info = biographies[list(biographies.keys())[predicted_speaker_index]]
                    st.session_state['speaker_name'] = list(biographies.keys())[predicted_speaker_index]
                    st.session_state['biography'] = speaker_info['biography']
                    st.session_state['image_urls'] = speaker_info['image_urls']
                    
                    st.session_state.page = 'result'
            else:
                st.error("Please upload an audio file.")
    elif st.session_state.page == 'result':
        st.write(f"Identified Speaker: {st.session_state['speaker_name']}")
        st.write("Biography:", st.session_state['biography'])
        if 'image_urls' in st.session_state:
            first_image_url = st.session_state['image_urls'][0]
            st.image(first_image_url, caption=st.session_state['speaker_name'], use_column_width=True, output_format='JPEG')
        if st.button("Identify Another Speaker"):
            st.session_state.page = 'upload'


main()
