import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_extras.let_it_rain import rain
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import base64
import random
import joblib

# Configuração inicial
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide"
)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.3f}'.format)
base_principal = pd.read_parquet('datasets/base_final_tratada.parquet')
modelo = joblib.load('datasets/modelo_cluster.pkl')

# Criando DataFrames
# Dados para a tabela
data = {
    "Coluna Original": [
        "AppID", "Name", "Release date", "Positive/Negative", "-", 
        "Price", "Header image", "Windows/Mac/Linux", "Achievements",
        "Recommendations", "Average playtime", "Median playtime",
        "Developers", "Publishers", "Categories/Genres/Tags", "-"
    ],
    "Coluna Transformada": [
        "ID_JOGO", "NOME", "ANO_LANCAMENTO", "INDICE_APROVACAO", "SCORE_POPULARIDADE",
        "PRECO", "IMAGEM_CAPA", "DISPONIVEL_[PLATAFORMA]", "CONQUISTAS",
        "RECOMENDACOES", "TEMPO_MEDIO_[PERÍODO]", "TEMPO_MEDIANO_[PERÍODO]",
        "DESENVOLVEDORES", "PUBLICADORAS", "CATEGORIAS/GENEROS/TAGS", "GENERO_PRINCIPAL"
    ],
    "Descrição": [
        "Identificador único", "Nome do jogo", "Ano de lançamento", 
        "Razão entre avaliações positivas/negativas (-1 se sem avaliações)",
        "INDICE_APROVAÇÃO × quantidade de avaliações",
        "Preço atual", "URL da imagem de capa", "Disponibilidade por sistema",
        "Número de conquistas", "Número de recomendações", "Tempo médio de jogo",
        "Tempo mediano de jogo", "Lista de desenvolvedores", "Lista de publicadoras",
        "Listas originais", "Gênero principal categorizado"
    ]
}
df = pd.DataFrame(data)

# Inicializa o DataFrame no session_state se não existir
def lista_sistemas(win, mac, linux):
    sistemas = ""
    if(win):
        sistemas += 'Windows'
    if(mac):
        if(len(sistemas) > 0):
            sistemas += ', '
        sistemas += 'MAC'
    if(linux):
        if(len(sistemas) > 0):
            sistemas += ', '
        sistemas += 'Linux'
    return sistemas

def calcular_top_jogos():
    filtros = st.session_state.filtros_aplicados
    # Gera o dataframe aplicando o modelo de ML para clusterizar o perfil do usuário
    TODOS_GENEROS = [
        'VIDEO 360', 'DOCUMENTÁRIO', 'EPISÓDIOS', 'FILME', 'CASUAL', 'CURTO', 
        'AÇÃO', 'AVENTURA', 'INDEPENDENTE', 'ESTRATÉGIA', 'MULTIJOGADOR MASSIVO',
        'UTILITÁRIOS', 'CORRIDA', 'SIMULAÇÃO', 'GRATUITO PARA JOGAR', 'RPG',
        'DESIGN E ILUSTRAÇÃO', 'ANIMAÇÃO E MODELAGEM', 'DESENVOLVIMENTO DE JOGOS',
        'EDUCAÇÃO', 'EDIÇÃO DE FOTOS', 'VIOLENTO', 'TREINAMENTO EM SOFTWARE',
        'ESPORTES', 'PRODUÇÃO DE ÁUDIO', 'PUBLICAÇÃO WEB', 'PRODUÇÃO DE VÍDEO',
        'CONTABILIDADE', 'ACESSO ANTECIPADO'
    ]

    novo_perfil = {genero: 0 for genero in TODOS_GENEROS}
    novo_perfil.update({
        'FAIXA_PRECO': filtros["FAIXA_PRECO"],
        'FAIXA_POPULARIDADE': filtros["FAIXA_POPULARIDADE"],
        'FAIXA_RECOMENDACAO': filtros["FAIXA_RECOMENDACAO"],
        'FAIXA_COLECIONAVEIS': filtros["FAIXA_COLECIONAVEIS"],
        'FAIXA_TEMPO_JOGO': filtros["FAIXA_TEMPO_JOGO"],
        **{genero: 1 for genero in filtros["GENERO"] if genero in filtros['GENERO']}
    })

    novo_perfil = pd.DataFrame([novo_perfil])

    cluster_usuario = modelo.predict(novo_perfil)[0]
    print(cluster_usuario)
    recomendacoes = base_principal[base_principal['CLUSTER_KMODES'] == cluster_usuario]
    
    # Filtra os jogos com base nos valores escolhidos pelo usuario
    mask = (
        (recomendacoes['PRECO'] >= filtros['PRECO_MIN']) & \
        (recomendacoes['PRECO'] <= filtros['PRECO_MAX']) & \
        (recomendacoes['ANO_LANCAMENTO'] >= filtros['ANO_MIN']) & \
        (recomendacoes['ANO_LANCAMENTO'] <= filtros['ANO_MAX']) & \
        (recomendacoes['DISPONIVEL_WINDOWS'] == filtros['WINDOWS']) & \
        (recomendacoes['DISPONIVEL_MAC'] == filtros['MAC']) & \
        (recomendacoes['DISPONIVEL_LINUX'] == filtros['LINUX'])
    )
    
    recomendacoes = recomendacoes[mask]
    
    # Cacular score de similaridade por cosseno
    df_similar = recomendacoes[['FAIXA_PRECO', 'FAIXA_POPULARIDADE', 'FAIXA_RECOMENDACAO', 'FAIXA_COLECIONAVEIS', 'FAIXA_TEMPO_JOGO', 'VIDEO 360', 'DOCUMENTÁRIO', 'EPISÓDIOS', 'FILME', 'CASUAL', 'CURTO', 'AÇÃO', 'AVENTURA', 'INDEPENDENTE', 'ESTRATÉGIA', 'MULTIJOGADOR MASSIVO', 'UTILITÁRIOS', 'CORRIDA', 'SIMULAÇÃO', 'GRATUITO PARA JOGAR', 'RPG', 'DESIGN E ILUSTRAÇÃO', 'ANIMAÇÃO E MODELAGEM', 'DESENVOLVIMENTO DE JOGOS', 'EDUCAÇÃO', 'EDIÇÃO DE FOTOS', 'VIOLENTO', 'TREINAMENTO EM SOFTWARE', 'ESPORTES', 'PRODUÇÃO DE ÁUDIO', 'PUBLICAÇÃO WEB', 'PRODUÇÃO DE VÍDEO', 'CONTABILIDADE', 'ACESSO ANTECIPADO']]
    colunas_escala_0a5 = [
        'FAIXA_PRECO',
        'FAIXA_POPULARIDADE',
        'FAIXA_RECOMENDACAO',
        'FAIXA_COLECIONAVEIS',
        'FAIXA_TEMPO_JOGO'
    ]

    X_norm = df_similar.copy()
    scaler = MinMaxScaler()
    X_norm[colunas_escala_0a5] = scaler.fit_transform(X_norm[colunas_escala_0a5])

    perfil = pd.DataFrame(novo_perfil, columns=X_norm.columns)
    perfil[colunas_escala_0a5] = scaler.transform(perfil[colunas_escala_0a5])
    
    # 2. Atribuição segura
    recomendacoes['SIMILARIDADE'] = cosine_similarity(X_norm, perfil).flatten()
    recomendacoes = recomendacoes.sort_values('SIMILARIDADE', ascending=False).reset_index(drop = True)
    
    # Ajustando informações
    recomendacoes['SISTEMAS_DISP'] = recomendacoes[['DISPONIVEL_WINDOWS', 'DISPONIVEL_MAC', 'DISPONIVEL_LINUX']].apply(lambda row: lista_sistemas(row['DISPONIVEL_WINDOWS'], row['DISPONIVEL_MAC'], row['DISPONIVEL_LINUX']), axis=1)
    recomendacoes['INDICE_APROVACAO'] = 10 * recomendacoes['INDICE_APROVACAO'].round(3)
    recomendacoes.loc[recomendacoes['INDICE_APROVACAO'] == -10, 'INDICE_APROVACAO'] = '-'
    
    return recomendacoes


if 'filtros_aplicados' not in st.session_state:
    st.session_state.filtros_aplicados = {"GENERO":["AÇÃO","RPG"],"FAIXA_POPULARIDADE":3,"FAIXA_RECOMENDACAO":3,"FAIXA_COLECIONAVEIS":3,"FAIXA_TEMPO_JOGO":3,"FAIXA_PRECO":3,"PRECO_MIN":0,"PRECO_MAX":60,"ANO_MIN":1997,"ANO_MAX":2025,"WINDOWS":True,"MAC":False,"LINUX":False}

if 'df_recomendados' not in st.session_state:
    st.session_state.df_recomendados = calcular_top_jogos()

if 'filtros_aplicados' not in st.session_state:
    st.session_state.filtros_aplicados = False

def define_faixa_preco(valor_min, valor_max):
    valor_medio = (valor_min + valor_max)/2
    if(valor_medio <= 0.99):
        return 1
    elif(valor_medio <= 2.99):
        return 2
    elif(valor_medio <= 5.59):
        return 3
    elif(valor_medio <= 9.99):
        return 4
    else:
        return 5
    
def gerar_emoji():
    valor =  random.randint(1, 5)
    if(valor == 1):
        return "🎈"
    elif(valor == 2):
        return "🏆"
    elif(valor == 3):
        valor = random.randint(1, 100)
        if(valor < 95):
            return "🎲"
        else:
            return "Parabéns, você descobriu um easter egg!"
    elif(valor == 4):
        return "🎮"
    elif(valor == 4):
        return "💡"
    elif(valor == 5):
        return "⚙️"
    
def image_to_base64(img_path):
    img = Image.open(img_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def image_viewer(img_base64):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .viewer-container {{
                width: 100%;
                height: 70vh;
                overflow: hidden;
                position: relative;
                border: 2px solid {STEAM_MEDIUM};
                border-radius: 8px;
                background-color: {STEAM_DARK};
            }}
            #zoom-image {{
                position: absolute;
                transition: transform 0.1s ease;
                cursor: grab;
                max-width: none;
                width: 100%;
            }}
            .controls {{
                position: absolute;
                right: 10px;
                z-index: 100;
            }}
            .zoom-btn {{
                background-color: {STEAM_ORANGE};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                margin: 0 5px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <div class="viewer-container" id="viewer">
            <img id="zoom-image" src="data:image/png;base64,{img_base64}" draggable="false">
        </div>
        <div class="controls">
            <button class="zoom-btn" onclick="zoomIn()">+</button>
            <button class="zoom-btn" onclick="zoomOut()">-</button>
            <button class="zoom-btn" onclick="resetZoom()">Reset</button>
        </div>
        

        <script>
            let scale = 1;
            let pos = {{ x: 0, y: 0 }};
            let isDragging = false;
            let startPos = {{ x: 0, y: 0 }};
            const img = document.getElementById('zoom-image');
            const container = document.getElementById('viewer');

            // Funções de zoom
            function zoomIn() {{
                scale *= 1.2;
                applyTransform();
            }}

            function zoomOut() {{
                scale /= 1.2;
                applyTransform();
            }}

            function resetZoom() {{
                scale = 1;
                pos = {{ x: 0, y: 0 }};
                applyTransform();
            }}

            function applyTransform() {{
                img.style.transform = `translate(${{pos.x}}px, ${{pos.y}}px) scale(${{scale}})`;
            }}

            // Controles de arraste
            img.addEventListener('mousedown', (e) => {{
                isDragging = true;
                startPos = {{ x: e.clientX - pos.x, y: e.clientY - pos.y }};
                img.style.cursor = 'grabbing';
            }});

            document.addEventListener('mousemove', (e) => {{
                if (!isDragging) return;
                pos.x = e.clientX - startPos.x;
                pos.y = e.clientY - startPos.y;
                applyTransform();
            }});

            document.addEventListener('mouseup', () => {{
                isDragging = false;
                img.style.cursor = 'grab';
            }});
        </script>
    </body>
    </html>
    """

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("css/style.css")  # Você precisará criar este arquivo CSS

# Cores da Steam para uso direto
STEAM_DARK = "#1B2838"
STEAM_MEDIUM = "#2A475E"
STEAM_LIGHT = "#66C0F4"
STEAM_ORANGE = "#F5AC27"
STEAM_GREEN = "#5BA32B"

# Configuração da barra lateral
with st.sidebar:
    # Cabeçalho estilizado
    st.markdown(f"""
    <div style='background: linear-gradient(to right, {STEAM_DARK}, {STEAM_MEDIUM});
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 25px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2)'>
        <h1 style='color:white; margin:0; display:flex; align-items:center;'>
            🎮 <span style='margin-left:10px;'>Steam Recommender</span>
        </h1>
        <p style='color:{STEAM_LIGHT}; margin:5px 0 0 0; font-size:14px;'>
            Personalize suas recomendações de jogos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gêneros com melhor espaçamento
    with st.sidebar.form("filtros_form"):
        st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
        genres = st.multiselect(
            "🎭 Selecione seus gêneros favoritos",
            ['VIDEO 360', 'DOCUMENTÁRIO', 'EPISÓDIOS', 'FILME', 'CASUAL', 'CURTO', 'AÇÃO', 'AVENTURA', 'INDEPENDENTE', 'ESTRATÉGIA', 'MULTIJOGADOR MASSIVO', 'UTILITÁRIOS', 'CORRIDA', 'SIMULAÇÃO', 'GRATUITO PARA JOGAR', 'RPG', 'DESIGN E ILUSTRAÇÃO', 'ANIMAÇÃO E MODELAGEM', 'DESENVOLVIMENTO DE JOGOS', 'EDUCAÇÃO', 'EDIÇÃO DE FOTOS', 'VIOLENTO', 'TREINAMENTO EM SOFTWARE', 'ESPORTES', 'PRODUÇÃO DE ÁUDIO', 'PUBLICAÇÃO WEB', 'PRODUÇÃO DE VÍDEO', 'CONTABILIDADE', 'ACESSO ANTECIPADO'],
            default=["AÇÃO", "RPG"],
            help="Escolha até 5 gêneros para focar sua busca"
        )
        
        # Sliders com ícones e melhor espaçamento
        st.markdown("<div style='margin-bottom:25px;'></div>", unsafe_allow_html=True)
        
        opcoes = {
            "Nenhuma": 0,
            "Baixíssima": 1,
            "Baixa": 2,
            "Média": 3,
            "Alta": 4,
            "Altíssima": 5
        }
        
        pop_label = st.select_slider(
            "📊 Popularidade:",
            options=list(opcoes.keys()),
            value="Média",
            help="Nível de popularidade dos jogos recomendados"
        )
        
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        
        recomenda_label = st.select_slider(
            "👍 Recomendações:",
            options=list(opcoes.keys()),
            value="Média",
            help="Nível de recomendações feitas"
        )
        
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        
        colecionavel_label = st.select_slider(
            "🏆 Colecionáveis:",
            options=list(opcoes.keys()),
            value="Média",
            help="Quantidade de itens colecionáveis disponíveis"
        )
        
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        
        tempo_label = st.select_slider(
            "⏱️ Tempo médio de jogo:",
            options=list(opcoes.keys()),
            value="Média",
            help="Duração média das sessões de jogo"
        )
        
        # Sliders numéricos com estilo
        st.markdown("<div style='margin-bottom:25px;'></div>", unsafe_allow_html=True)
        
        price_range = st.slider(
            "💰 Faixa de preço (USD)",
            0, 1000, (0, 60),
            help="Intervalo de preço dos jogos recomendados"
        )
        
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        
        ano_range = st.slider(
            "📅 Ano de lançamento",
            1997, 2025, (1997, 2025),
            help="Período de lançamento dos jogos"
        )
        
        # Checkboxes em formato de cards
        st.markdown("<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True)
        st.markdown("💻 **Sistemas operacionais:**", unsafe_allow_html=True)
        
        cols = st.columns(3)
        with cols[0]:
            op1 = st.checkbox("Windows", True, key="win", 
                            help="Jogos compatíveis com Windows")
        with cols[1]:
            op2 = st.checkbox("MAC", False, key="mac",
                            help="Jogos compatíveis com macOS")
        with cols[2]:
            op3 = st.checkbox("Linux", False, key="linux",
                            help="Jogos compatíveis com Linux")

        submitted = st.form_submit_button("🚀 Gerar Recomendações", type="primary", help="Clique para gerar suas recomendações personalizadas")

    if submitted:
        with st.spinner('Processando recomendações...'):
            # Obter todos os valores dos filtros
            st.session_state.filtros_aplicados = {
                "GENERO": genres,
                "FAIXA_POPULARIDADE": opcoes[pop_label],
                "FAIXA_RECOMENDACAO": opcoes[recomenda_label],
                "FAIXA_COLECIONAVEIS": opcoes[colecionavel_label],
                "FAIXA_TEMPO_JOGO": opcoes[tempo_label],
                "FAIXA_PRECO": define_faixa_preco(price_range[0], price_range[1]),
                "PRECO_MIN": price_range[0],
                "PRECO_MAX": price_range[1],
                "ANO_MIN": ano_range[0],
                "ANO_MAX": ano_range[1],
                "WINDOWS": op1,
                "MAC": op2,
                "LINUX": op3
            }
            
            # Efeito visual
            # st.balloons()
            rain(
                emoji=gerar_emoji(),
                font_size=35,
                falling_speed=5,
                animation_length=0.25  
            )

            st.session_state.df_recomendados = calcular_top_jogos()
            
            # Mostrar os filtros aplicados
            st.success("✅ Filtros aplicados com sucesso!")

# HTML para cabeçalho no estilo Steam
st.markdown(f"""
<div style='background: linear-gradient(to right, {STEAM_DARK}, {STEAM_MEDIUM}); box-shadow: 0 4px 8px rgba(0,0,0,0.2); padding:20px; border-radius:5px; margin-bottom:20px;'>
    <h1 style='color:{STEAM_LIGHT}; text-align:center;'>STEAM GAME RECOMMENDER</h1>
    <p style='color:white; text-align:center;'>Encontre seu próximo jogo favorito na Steam!</p>

</div>
""", unsafe_allow_html=True)

# HTML para cabeçalho no estilo Steam
st.markdown(f"""
<div style='background-color:{STEAM_DARK}; padding:20px; border-radius:5px; margin-bottom:20px;'>
    <p style='color:white; text-align:center;'> Este é um dash interativo cujo o objetivo é te dar recomendações de jogos na plataforma da <a href='https://store.steampowered.com/?l=portuguese'>Steam</a>, a maior plataforma de jogos para computador.</p>
    <ul style='color:white; padding-left: 5rem;'>
        <li>Use a barra lateral para filtrar os jogos de acordo com suas preferências e receba na aba '🎯 Principais recomendações' os top 3  jogos mais recomendados para você!</li>
        <li>Para ver mais recomendações com base em suas preferências, use a aba '🌟 + Jogos Recomendados'.</li>
        <li>Para entender mais sobre os dados do dashboard, as análises por detrás dos panos e o sistema de recomedações, vá para a aba "🎲 Sobre o Dashboard".</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# HTML para cabeçalho no estilo Steam
st.markdown(f"""
<div style='background-color:{STEAM_DARK}; padding:20px; border-radius:5px; margin-bottom:20px;'>
    <p style='color:white; text-align:center;'>⚠️ Atenção: esse dashboard selecionou apenas jogos marcados como livres para todos os públicos na steam</p>
</div>
""", unsafe_allow_html=True)

# Abas para diferentes seções
tab1, tab2, tab3 = st.tabs(["🎯 Principais Recomendações", "🌟 + Jogos Recomendados",  "🎲 Sobre o Dashboard"])

with tab1:
    df_top_3 = st.session_state.df_recomendados.head(3)
    # Div com cor de fundo diferente
    st.markdown(f"""
    <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
        <h2 style='color:{STEAM_LIGHT};'>Recomendações Personalizadas</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Painel de jogos recomendados (você pode adicionar mais)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nota = df_top_3.iloc[0]['INDICE_APROVACAO']
        nota = nota if nota == '-' else '{:.2f}'.format(float(nota))
            
        st.markdown(f"""
        <div style='background-color:#2A475E; padding:10px; border-radius:5px; text-align:center;'>
            <img src="{df_top_3.iloc[0]['IMAGEM_CAPA']}" width='100%' style='border-radius:5px;'>
            <h3 style='color:#66C0F4;'>{df_top_3.iloc[0]['NOME']}</h3>
            <p style='color:white;'>{df_top_3.iloc[0]['GENEROS']}</p>
            <p style='color:#F5AC27;'>⭐ {nota}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        nota = df_top_3.iloc[0]['INDICE_APROVACAO']
        nota = nota if nota == '-' else '{:.2f}'.format(float(nota))
        
        st.markdown(f"""
        <div style='background-color:#2A475E; padding:10px; border-radius:5px; text-align:center;'>
            <img src="{df_top_3.iloc[1]['IMAGEM_CAPA']}" width='100%' style='border-radius:5px;'>
            <h3 style='color:#66C0F4;'>{df_top_3.iloc[1]['NOME']}</h3>
            <p style='color:white;'>{df_top_3.iloc[1]['GENEROS']}</p>
            <p style='color:#F5AC27;'>⭐ {nota}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        nota = df_top_3.iloc[0]['INDICE_APROVACAO']
        nota = nota if nota == '-' else '{:.2f}'.format(float(nota))
        
        st.markdown(f"""
        <div style='background-color:#2A475E; padding:10px; border-radius:5px; text-align:center;'>
            <img src="{df_top_3.iloc[2]['IMAGEM_CAPA']}" width='100%' style='border-radius:5px;'>
            <h3 style='color:#66C0F4;'>{df_top_3.iloc[2]['NOME']}</h3>
            <p style='color:white;'>{df_top_3.iloc[2]['GENEROS']}</p>
            <p style='color:#F5AC27;'>⭐ {nota}/10</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown(f"""
    <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
        <h2 style='color:{STEAM_LIGHT};'>Top Jogos Recomendados da Steam</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabela de top jogos
    df_top = st.session_state.df_recomendados.reset_index()
    df_top = df_top[['index', 'NOME', 'INDICE_APROVACAO', 'GENERO_PRINCIPAL', 'ANO_LANCAMENTO', 'PRECO', 'SISTEMAS_DISP', 'DESENVOLVEDORES', 'TAGS']]
    df_top = df_top.rename(columns={'index': 'Posição',
                           'NOME': 'Jogo',
                           'INDICE_APROVACAO': 'Avaliação',
                           'GENERO_PRINCIPAL': 'Gênero principal',
                           'TAGS': 'Tags',
                           'ANO_LANCAMENTO': 'Ano de lançamento',
                           'PRECO': 'Preço',
                           'SISTEMAS_DISP': 'Sistemas',
                           'DESENVOLVEDORES': 'Desenvolvedores'
                           })
    
    st.dataframe(
        df_top.style.format({
            'Avaliação': lambda x: x if x == '-' else '{:.2f}'.format(float(x)),
            'Preço': '${:.2f}'
        }).applymap(lambda x: f"color: {STEAM_LIGHT}", subset=["Jogo"])
        .applymap(lambda x: f"color: {STEAM_ORANGE}", subset=["Avaliação"]),
        hide_index=True,
        use_container_width=True
    )
    
with tab3:

    st.markdown(f"""
    <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
        <h2 style='color:{STEAM_LIGHT};'>Sobre a base de dados</h2>
        <p>Atualmente, sistemas de recomendação estão presentes em praticamente todas as plataformas digitais — da Netflix ao Spotify, passando por Amazon e YouTube — ajudando usuários a descobrirem conteúdos relevantes de forma personalizada. Inspirado por esse cenário, este dashboard propõe o desenvolvimento de um sistema de recomendação de jogos focado na Steam, uma das maiores plataformas de distribuição de jogos para computador. O objetivo é oferecer sugestões inteligentes de títulos com classificação livre, facilitando a descoberta de novos jogos por parte do público geral, com base em características dos próprios games e nas preferências indicadas pelos usuários.</p>   
        <p>Para a base de dados, usei o data set <a href="https://www.kaggle.com/datasets/fronkongames/steam-games-dataset/data">Steam Games Dataset</a> disponível no Kaggle. Ele reúne informações diversas como "Nome", "Descição", "Preço", "Data de lançamento", "Avaliações", "Recomendações" e muito mais de mais de 110 mil jogos publicados na Steam, coletadas pela API da própria plataforma.</p>
        <p>Confira abaixo como os dados foram organizados!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar e converter imagem
    img_base64 = image_to_base64("img/Fluxograma_limpeza.png")
    
    # Container Steam
    with st.container():
        st.markdown(f"""
        <div style='
            background-color: {STEAM_DARK};
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-left: 4px solid {STEAM_ORANGE};
        '>
            <h3 style='color:{STEAM_LIGHT};'>Fluxograma de Tratamento de Dados</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizador de Imagem
        st.components.v1.html(image_viewer(img_base64), height=600)
        
        st.markdown(f"""
        <div style='color:{STEAM_LIGHT}; font-size:0.9em; text-align: center; margin-top: -150px'>
            🔍 Clique nos botões para zoom | 🖱️ Arraste para navegar
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown(f"""
        <div style='
            background-color: {STEAM_DARK};
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            border-left: 4px solid {STEAM_ORANGE};
        '>
            <h3 style='color:{STEAM_LIGHT};'>Estrutura da Base de Dados</h3>
            <p style='color:white;'>Sendo exaustivo e explicando mais detalhadamento o passo a passo do fluxograma, como haviam várias variáveis iniciais, o primeiro passo foi filtrar as informações que não faziam sentido para o propósito do dashboard e tratar os dados. Portanto, seguem os tratamentos utilizados:</p>
            <p style='color:white;'>Filtrei os jogos impróprios e os que não tinham a informação de gênero e transformei as colunas originais em colunas mais analítico. O objetivo disso é preparar os dados para o Machine Learning aplicado.</p>
            <p style='color:white; font-size: 14px'>~ As colunas originais que não aparecem foram removidas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabela principal
        st.dataframe(
            df,
            column_config={
                "Coluna Original": "Original",
                "Coluna Transformada": "Transformada",
                "Descrição": st.column_config.Column(
                    "Descrição",
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        st.markdown(f"""
        <div style='
            background-color: {STEAM_MEDIUM};
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        '>
            <p style='color:white;'>Feito isso e pensando em um modelo de recomendação de jogos, o ideal é ter as categorias bem definidas. Portanto, criei várias colunas para cada gênero de jogo através do One-Hot Encoded.</p>
            <div style='grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;'>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin-bottom: 5px;'><strong>Gêneros:</strong></p>
                    <p style='color:white;'>VIDEO 360, DOCUMENTÁRIO, EPISÓDIOS, FILME, CASUAL, CURTO, AÇÃO, AVENTURA, INDEPENDENTE, ESTRATÉGIA, MULTIJOGADOR MASSIVO, UTILITÁRIOS, CORRIDA, SIMULAÇÃO, GRATUITO PARA JOGAR, RPG, DESIGN E ILUSTRAÇÃO, ANIMAÇÃO E MODELAGEM, DESENVOLVIMENTO DE JOGOS, EDUCAÇÃO, EDIÇÃO DE FOTOS, VIOLENTO, TREINAMENTO EM SOFTWARE, ESPORTES, PRODUÇÃO DE ÁUDIO, PUBLICAÇÃO WEB, PRODUÇÃO DE VÍDEO, CONTABILIDADE, ACESSO ANTECIPADO</p>
                </div>
            </div>
            <p style='color:white;'>Por fim, como muitos valores eram quantitativos e variávam em escala, foi necessário criar métricas normalizadas que dividissem bem os dados. Daí surgiram as colunas de faixas, que quebram as variáveis descritas abaixo em quintis</p>
            <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;'>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin: 0;'>FAIXA_PRECO</p>
                    <p style='color:white; margin: 0; font-size: 14px;'>Classificação em 5 níveis (1 a 5)</p>
                </div>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin: 0;'>FAIXA_POPULARIDADE</p>
                    <p style='color:white; margin: 0; font-size: 14px;'>Classificação em 5 níveis (1 a 5) + 0 para indicar que não houveram avaliações</p>
                </div>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin: 0;'>FAIXA_RECOMENDACAO</p>
                    <p style='color:white; margin: 0; font-size: 14px;'>Classificação em 5 níveis (1 a 5) + 0 para indicar que não houveram recomendações</p>
                </div>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin: 0;'>FAIXA_COLECIONAVEIS</p>
                    <p style='color:white; margin: 0; font-size: 14px;'>Classificação em 5 níveis (1 a 5) + 0 para indicar que não possui colecionáveis</p>
                </div>
                <div style='background-color: {STEAM_DARK}; padding: 10px; border-radius: 4px;'>
                    <p style='color:{STEAM_ORANGE}; margin: 0;'>FAIXA_TEMPO_JOGO</p>
                    <p style='color:white; margin: 0; font-size: 14px;'>Classificação em 5 níveis (1 a 5) + 0 para indicar que ninguém jogou</p>
                </div>
            </div>
            <p style='color:white;'>Abaixo é possível ver como os dados vieram (dados brutos), bem como eles ficaram após o tratamento de quintis (dados tratados). Também trago a correlação das variáveis e dispersão que os dados apresentavam.</p>
        </div>
        """, unsafe_allow_html=True)
        
        
    # Abas para diferentes seções
    tab_brutos, tab_tratados, tab_corr = st.tabs(["🎲 Dados brutos (análise exploratória)", "⚙️ Como ficaram os dados tratados?", "📊 Correlação das variáveis e dispersão dos dados"])

    with tab_brutos:
        # Div com cor de fundo diferente
        st.markdown(f"""
        <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
            <h4 style='color:{STEAM_LIGHT};'>Distribuição das variáveis iniciais de interesse, sem corte por quartil</h4>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.image("img/preco_antes.png", caption="Distribuição do preço", use_container_width=True)
        with col2:
            st.image("img/recomendacoes_antes.png", caption="Distribuição das recomendações", use_container_width=True)
        with col3:
            st.image("img/tempo_medio_antes.png", caption="Distribuição do tempo médio de jogo", use_container_width=True)
        with col4:
            st.image("img/popularidade_antes.png", caption="Distribuição da popularidade", use_container_width=True)
        with col5:
            st.image("img/colecionaveis_antes.png", caption="Distribuição de colecionários", use_container_width=True)
        with col6:
            st.image("img/aprovacao_antes.png", caption="Dsitribuição da aprovação", use_container_width=True)
        with col7:
            st.image("img/ano_lancamento_antes_depois.png", caption="Distribuição do anomes de lançamento", use_container_width=True)

    with tab_tratados:
        # Div com cor de fundo diferente
        st.markdown(f"""
        <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
            <h4 style='color:{STEAM_LIGHT};'>Distribuição das variáveis tratadas, com corte por quartil</h4>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image("img/preco_depois.png", caption="Distribuição do preço", use_container_width=True)
        with col2:
            st.image("img/recomendacoes_depois.png", caption="Distribuição das recomendações", use_container_width=True)
        with col3:
            st.image("img/tempo_medio_depois.png", caption="Distribuição do tempo médio de jogo", use_container_width=True)
        with col4:
            st.image("img/popularidade_depois.png", caption="Distribuição da popularidade", use_container_width=True)
        with col5:
            st.image("img/colecionaveis_depois.png", caption="Distribuição de colecionários", use_container_width=True)

        st.markdown(f"""
        <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
            <p style='color:white;'>Observe que a variável ANO_LANCAMENTO e APROVAÇÃO não aparecem aqui. Isso acontece porque elas estão mais bem distribuidas e não foram utilizadas na clusterização do sistema de recomendações.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_corr:
        # Div com cor de fundo diferente
        st.markdown(f"""
        <div style='background-color:{STEAM_MEDIUM}; padding:15px; border-radius:5px; margin-bottom:20px;'>
            <h4 style='color:{STEAM_LIGHT};'>Correlação dos dados</h4>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)
        with col1:
            st.image("img/correlacao.png", caption="Matriz de corrleção das variáveis", use_container_width=True)
        with col2:
            st.image("img/dispercao_dados.png", caption="Disperção dos dados", use_container_width=True)

    st.markdown(f"""
    <div style='
        background-color: {STEAM_DARK};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border-left: 4px solid {STEAM_ORANGE};
    '>
        <h3 style='color:{STEAM_LIGHT};'>Partindo para o modelo de recomendação de jogos</h3>
        <p style='color:white;'>Com os dados já tratados e a base final consolidada para o ML, precisamos entender onde estamos saíndo e para onde queremos chegar. Para um sistema de recomendação de jogos, é preciso observar quais jogos possuem características semelhantes com base nos dados apresentados e, dadas as preferências do jogador, mostrar games que possuem estas características.</p>
        <p style='color:white;'>Portanto, neste caso uma boa ideia é usar algum modelo de aprendizado não supervisionados, que nos retornam clusters (agrupamentos) de jogos que possuem perfils semelhantes. Por isso, foi necessário tratar os dados previamente, bem como realizar a quebra por quantils. Desta forma, conseguimos enxergar melhor o que caracteriza um jogo e seus semelhantes.</p>
        <p style='color:white;'>Para essa clusterização, usei as variáveis 'FAIXA' e de Generos (One Hot Encoded) apresentadas anteriormente. Assim, cada jogo tem uma faixa de preço, popularidade, recomendação de outros players, colecionáveis, tempo de jogo e o número 1 nas colunas dos generos que ele possui (o valor fica 0 caso não tenha aquele gênero). Assim, é possível realizar a clusterização.</p>
        <p style='color:white;'>Como existem vários jogos selecionados para o modelo (+80.000 mesmo depois de aplicar os filtros), é provável que nossa clusterização apresente vários grupos distintos. Então usei o <a href='https://en-m-wikipedia-org.translate.goog/wiki/Elbow_method_(clustering)?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt&_x_tr_pto=tc'>método do cotovelo</a> para avaliar a faixa ideial de clusters, como mostrado a seguir:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("img/cotovelo.png", caption="Matriz de corrleção das variáveis", use_container_width=True)
    with col2:
        st.image("img/cotovelo_regressao.png", caption="Disperção dos dados", use_container_width=True)
        
    st.markdown(f"""
    <div style='
        background-color: {STEAM_DARK};
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border-left: 4px solid {STEAM_ORANGE};
    '>
        <p style='color:white;'>Nestes gráficos é possível ver que a quantidade de clusters utilizada cresce muito para que o ganho seja bom. Isso também implica na demora maior do algoritmo para rodar novos pontos. Sendo assim, fiz uma regressão não linear simples para entender melhor a tendência dos meus dados e definir a quantidade de 360 clusters para o sistema de recomendações.</p>
        <p style='color:white;'>⚠️ Ponto importante! Pelo conjunto de dados diversificado, o ideal é utilizar o K-Prototypes, que consegue clusterizar variáveis numéricas (as faixas) e categorias (as de One Hot Encoded) ao mesmo tempo. No entanto, a máquina utilizada era limitada a 16 Gb de RAM, o que tornou bem custoso e demorado o processamento. Daí veio a ideia de usar o K-Modes, que funciona bem para variáveis categorias, considerando que os valores de "Faixa" também são categóricos. Não é matematicamente ideal usar K-Modes com variáveis ordinais discretizadas pois o algoritmo ignora a ordem e magnitude relativa das diferentes faixas, mas optei pela simplicidade no modelo e segui desta forma.</p>
        <p style='color:white;'>🧠 Nesse sentido, para equilibrar esse ponto de alerta, melhorei o algoritmo de recomendação usando a <a href="https://pt.wikipedia.org/wiki/Similaridade_por_cosseno">similaridade por cosseno</a>. Então, assim que o jogador escolher os filtros desejados, o algoritmo de ML prevê o cluster que mais se adequa as informações, retornando uma lista de jogos recomendados. Daí, normalizo as variáveis de faixa entre 0 e 1 e aplico a similaridade por cossenos para escolher o top 3 de jogos mais próximos ao que o usuário deseja. Essa similaridade retorna um valor entre -1 a 1 (no nosso caso um valor entre 0 e 1 pois todos os valores são positivos), indicando quão bem os vetores comparados são próximos entre si. O retorno mais próximo de 1 indica similaridade e mais próximo de -1, indica dissimilaridade (diferença).</p>
        <p style='color:white;'>🎯 Por ser um algoritmo de aprendizado não supervisionado e ter uma imensa variedade de clusters, é difícil dizer com precisão a assertividade do modelo. Por meio da média, é possível traçado um perfil médio para cada cluster, mas como são muitas variáveis categóricas de gênero, alguns valores acabam não fazendo tanto sentido, até mesmo para o PCA, que só funciona bem se todas as colunas forem numéricas. Como este é um sistema de recomendação, uma boa maneira de testar é na prática, colocando os filtros que deseja e observando se os jogos condizem com os filtros esperados.</p>
        <p style='color:white;'><b>Concluindo:</b> apesar do modelo apresentar limitações e ausência de interações reais do usuário, ele se mostrou funcional e promissor, servindo de base para recomendações personalizadas.</p>
        <p style='color:white;'>🎮 Dito tudo isso, escolha os filtros que desejar e veja o algoritmo em ação recomendando os melhores jogos para você!</p>
        <p style='color:white; font-size: 14px;'>Nota: se quiser entender mais sobre K-Means, K-Modes e K-Prototype, recomendo o seguinte artigo: <a href='https://medium.com/@reddyyashu20/k-means-kmodes-and-k-prototype-76537d84a669'>link</a></p> 
    </div>
    """, unsafe_allow_html=True)

# Rodapé no estilo Steam
st.markdown(f"""
<div style='background-color:{STEAM_DARK}; padding:10px; border-radius:5px; margin-top:30px; text-align:center;'>
    <p style='color:{STEAM_LIGHT};'>Steam Game Recommender by Felipe Maia Lopes Sinoti © 2025 - Todos os direitos reservados</p>
</div>
""", unsafe_allow_html=True)
