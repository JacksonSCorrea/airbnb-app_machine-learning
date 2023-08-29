import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Importando os objetos
with open('objects.pkl', 'rb') as zip_model:
    kmeans, pca, encoder, scaler, \
    encoder_0, encoder_1, encoder_2, encoder_3, encoder_4, encoder_5, \
    scaler_0, scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, \
    modelo_0, modelo_1, modelo_2, modelo_3, modelo_4, modelo_5 = pickle.load(zip_model)

# =====================================================================================================================
# STREAMLIT

st.title("AirPrice")
st.subheader("Descubra o preço ideal da diária de um imóvel do Airbnb Califórnia!")
st.write("Desenvolvido por Jackson Corrêa - v00 - agosto/2023 |  <a href='https://www.linkedin.com/in/jackson-corrêa' target='_blank'>Acesse meu LinkedIn</a>  |  <a href='https://www.github.com/JacksonSCorrea' target='_blank'>Acesse meu GitHub</a>" , unsafe_allow_html=True)

st.subheader("\n\nFormulário de Entrada de Dados")

# Variáveis para coletar os dados
neighbourhood_cleansed = st.selectbox("Bairro",
    ['Mission', 'Marina', 'Bernal Heights', 'Downtown/Civic Center',
    'South of Market', 'Haight Ashbury', 'West of Twin Peaks',
    'North Beach', 'Castro/Upper Market', 'Inner Richmond',
    'Potrero Hill', 'Russian Hill', 'Outer Mission',
    'Western Addition', 'Nob Hill', 'Noe Valley', 'Excelsior',
    'Glen Park', 'Outer Sunset', 'Outer Richmond', 'Bayview',
    'Inner Sunset', 'Twin Peaks', 'Lakeshore', 'Parkside',
    'Visitacion Valley', 'Crocker Amazon', 'Ocean View',
    'Presidio Heights', 'Seacliff', 'Diamond Heights', 'Presidio',
    'Chinatown', 'Pacific Heights', 'Financial District'])

property_type = st.selectbox("Tipo de Propriedade",
    ['Condominium', 'Guest suite', 'Guesthouse', 'Cottage', 'Hostel',
    'Loft', 'Boat', 'Serviced apartment', 'Townhouse', 'House',
    'Bed and breakfast', 'Boutique hotel', 'Bungalow', 'Other',
    'Hotel', 'Villa', 'Tiny house', 'Aparthotel'])

room_type = st.selectbox("Tipo de Quarto", ["Private room", "Entire home/apt", "Shared room"])

accommodates = st.number_input("Acomodações", min_value=1, step=1)

bathrooms = st.number_input("Banheiros", min_value=0, step=1, value=1)

bedrooms = st.number_input("Quartos", min_value=0, step=1, value=1)

beds = st.number_input("Camas", min_value=0, step=1, value=1)

bed_type = st.selectbox("Tipo de Cama", ["Real Bed", "Pull-out Sofa", "Futon", "Airbed", "Couch"])

minimum_nights = st.number_input("Mínimo de Noites", min_value=1, step=1)

cancellation_policy = st.selectbox("Política de Cancelamento", ["Moderate", "Strict 14 with grace period", "Flexible"])
if cancellation_policy == "Moderate":
    cancellation_policy='moderate'
elif cancellation_policy=="Strict 14 with grace period":
    cancellation_policy='strict_14_with_grace_period'
else:
    cancellation_policy='flexible'

instant_bookable = st.radio("Reserva Imediata", ["Yes", "No"])
if instant_bookable=="Yes":
    instant_bookable = 't'  #true
else:
    instant_bookable = 'f'  #false


# inserindo um botão na tela
btn_predict = st.button("Descobrir preço")


# verifica se o botão foi acionado
if btn_predict:

    entradas = {'cancellation_policy':cancellation_policy,
                'instant_bookable' :instant_bookable,
                'neighbourhood_cleansed' :neighbourhood_cleansed,
                'property_type' :property_type,
                'room_type' : room_type,
                'accommodates' : accommodates,
                'bathrooms' : bathrooms,
                'bedrooms' : bedrooms,
                'beds' : beds,
                'bed_type' : bed_type,
                'minimum_nights' : minimum_nights }


    # entradas = {'cancellation_policy':'moderate',
    #             'instant_bookable' :'f',
    #             'neighbourhood_cleansed' :'Bernal Heights',
    #             'property_type' :'Hostel',
    #             'room_type' : 'Private room',
    #             'accommodates' : 6,
    #             'bathrooms' : 1,
    #             'bedrooms' : 1,
    #             'beds' : 3,
    #             'bed_type' : 'Pull-out Sofa',
    #             'minimum_nights' : 15 }

    df = pd.DataFrame([entradas])

    df_entradas = df.copy()

    # Aplicando o encoder
    df_entradas = encoder.transform(df_entradas)

    # Aplicando o scaler
    df_entradas = scaler.transform(df_entradas)

    # Aplicando o pca
    df_entradas_cluster = pca.transform(df_entradas)

    # Aplicando o kmeans para saber qual o cluster previsto
    rotulo = kmeans.predict(df_entradas_cluster)

    # printando o rótulo do cluster
    st.subheader(f'O imóvel pertence ao grupo {rotulo[0]+1}')

    # Restaurando o dataset depois de ter sido encodado e escalado
    df_entradas = df.copy() 

    if rotulo == 0:
        df_entradas = encoder_0.transform(df_entradas)
        df_entradas = scaler_0.transform(df_entradas)
        preco = modelo_0.predict(df_entradas)
        st.write('As características que mais influenciam o preço dos imóveis deste grupo são:')
        st.write('• Tipo de quarto: "Private rooms" tendem a ter um desempenho melhor na quantidade de locações')
        st.write('• Acomodações: no geral, quartos com poucas acomodações possuem melhor desempenho')
        st.write('• Quantidade de quartos: no geral, os imóveis com melhor deempenho possuem somente um quarto')

    if rotulo == 1:
        df_entradas = encoder_1.transform(df_entradas)
        df_entradas = scaler_1.transform(df_entradas)
        preco = modelo_1.predict(df_entradas)
        st.write('As características que mais influenciam o preço dos imóveis deste grupo são:')
        st.write('• Acomodações: no geral, quartos com poucas acomodações possuem melhor desempenho')
        st.write('• Quantidade de quartos: no geral, os imóveis com melhor deempenho possuem somente um quarto')
        st.write('• Tipo de quarto: "Private rooms" tendem a ter um desempenho melhor na quantidade de locações')

    if rotulo == 2:
        df_entradas = encoder_2.transform(df_entradas)
        df_entradas = scaler_2.transform(df_entradas)
        preco = modelo_2.predict(df_entradas)
        st.write('A característica que mais influencia o preço dos imóveis deste grupo são:')
        st.write('• Acomodações: dentro de um grupo de imóveis performáticos, a maioria possui 2 acomodações')

    if rotulo == 3:
        df_entradas = encoder_3.transform(df_entradas)
        df_entradas = scaler_3.transform(df_entradas)
        preco = modelo_3.predict(df_entradas)
        st.write('A característica que mais influencia o preço dos imóveis deste grupo são:')
        st.write('• Quantidade de quartos: dentro de um grupo de imóveis performáticos, a maioria possui 1 quarto')

    if rotulo == 4:
        df_entradas = encoder_4.transform(df_entradas)
        df_entradas = scaler_4.transform(df_entradas)
        preco = modelo_4.predict(df_entradas)
        st.write('A característica que mais influencia o preço dos imóveis deste grupo são:')
        st.write('• Quantidade de quartos: dentro de um grupo de imóveis performáticos, a maioria possui 1 quarto')

    if rotulo == 5:
        df_entradas = encoder_5.transform(df_entradas)
        df_entradas = scaler_5.transform(df_entradas)
        preco = modelo_5.predict(df_entradas)
        st.write('A característica que mais influencia o preço dos imóveis deste grupo são:')
        st.write('• Quantidade de quartos: dentro de um grupo de imóveis performáticos, a maioria possui 1 quarto')

    # Exibindo o valor do preço
    st.subheader(f'Preço sugerido: U$ {round(float(preco[0]),2)}') #exibe
