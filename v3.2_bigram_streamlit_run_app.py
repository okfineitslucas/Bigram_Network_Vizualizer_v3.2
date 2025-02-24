import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from nltk.util import bigrams
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string
import spacy
from io import BytesIO, StringIO
import math
import time
import adjustText  # pip install adjustText
import os
os.system("pip install pygraphviz")
import pygraphviz  # Now it should work


# ---------------------------
# 1) Streamlit Layout Config
# ---------------------------
st.set_page_config(layout="wide")

# ---------------------------
# 2) Title & Intro
# ---------------------------
st.markdown("""
# 🚀 Bigram Network Visualization (Python)
Analyze co-occurring words in **two** layout styles:
1. **Radial Web (sfdp)** – approximate force-directed “hub/spoke”
2. **Hierarchical Flow (dot)** – top-down or left-to-right  

**How to Use**:
1. Upload a CSV/Excel file with text data.  
2. Pick how many terms & min bigram frequency.  
3. Preview data, then click **Generate** to see your chosen layout.  

**Now** with BFS-based subgraphs, sfdp radial or hierarchical flow, minimized label collisions, and a more cohesive structure.  

*Credit: Built by [@okfineitslucas](https://github.com/okfineitslucas)*
""")

# ---------------------------
# 3) CSS for Buttons
# ---------------------------
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #e05757 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    height: 3em !important;
    width: 20em !important;
    font-size: 1.2em !important;
    font-weight: 600 !important;
    margin: 0.5em 0px !important;
    transition: background-color 0.3s ease !important;
}
div.stButton > button:first-child:hover {
    background-color: #ff6666 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 4) NLTK Setup
# ---------------------------
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# ---------------------------
# 5) spaCy Model
# ---------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser","tagger"])

# ---------------------------
# 6) Stopwords & Preprocessing
# ---------------------------
BASE_STOPWORDS = {
    "the","and","is","in","to","of","for","on","with","at","by","from","that","this","it","a","an","be",
    "are","as","was","were","has","have","or","but","not","about","after","again","all","am","any","because",
    "been","before","being","between","both","can","did","do","does","doing","down","during","each","few",
    "he","her","here","him","his","how","if","into","its","me","more","most","my","no","now","only","other",
    "our","out","over","same","she","so","some","such","than","their","them","then","there","these","they",
    "through","under","until","up","very","we","what","when","where","which","while","who","whom","why",
    "will","you","your","rt","one","get","just","like","see","new","time","make","good","know","day","want",
    "go","pic","www","imdb","team","star","via","amp","n’t","’re","…"
}

def remove_urls(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"pic\.twitter\.com/\S+", "", text)
    return text

def strip_emojis_keep_mentions_hashtags(text: str) -> str:
    # Remove emojis but keep # and @
    return text.encode("ascii","ignore").decode()

def remove_punct_except_symbols(text: str) -> str:
    # Keep #, @, and ' 
    keepers = ["#", "@", "'"]
    punc = string.punctuation
    for k in keepers:
        punc = punc.replace(k, "")
    return text.translate(str.maketrans('', '', punc))

def is_meaningful_token(token: str) -> bool:
    if len(token) < 2:
        return False
    if token.lower() in BASE_STOPWORDS:
        return False
    if token.isnumeric():
        return False
    return True

def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = strip_emojis_keep_mentions_hashtags(text)
    text = remove_punct_except_symbols(text)
    return text

def tokenize_and_filter(doc, ner_enabled=True):
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        # Keep named entities if NER is on
        if ner_enabled and token.ent_type_ in ["PERSON","ORG","GPE","LOC"]:
            tokens.append(lemma)
        else:
            if is_meaningful_token(lemma):
                tokens.append(lemma)
    return tokens

# ---------------------------
# 7) Bigram Extraction
# ---------------------------
from nltk.util import bigrams

def generate_bigrams_from_texts(texts, ner_enabled=True, batch_size=500, update_every=200):
    start_time = time.time()
    total = len(texts)
    done = 0
    prog_bar = st.progress(0)
    status_text = st.empty()

    global nlp
    # Toggle NER
    if ner_enabled:
        if "ner" not in nlp.pipe_names:
            nlp.add_pipe("ner", last=True)
    else:
        if "ner" in nlp.pipe_names:
            nlp.remove_pipe("ner")

    # Preprocess
    prepped_texts = [preprocess_text(t) for t in texts]

    for i, doc in enumerate(nlp.pipe(prepped_texts, batch_size=batch_size), start=1):
        tokens = tokenize_and_filter(doc, ner_enabled=ner_enabled)
        for bg in bigrams(tokens):
            yield bg
        done += 1

        if done % update_every == 0 or done == total:
            elapsed = time.time() - start_time
            docs_per_sec = done / elapsed if elapsed else 1
            eta = (total - done) / docs_per_sec
            prog_bar.progress(done / total)
            status_text.write(
                f"Processed {done}/{total} docs. "
                f"Elapsed: {elapsed:.1f}s. "
                f"ETA: {eta:.1f}s."
            )

import pandas as pd
import numpy as np
import networkx as nx

def read_data_in_chunks(file, chunk_size=20000):
    for chunk in pd.read_csv(file, chunksize=chunk_size, encoding='utf-8', sep=None, engine='python'):
        yield chunk

@st.cache_data
def compute_bigrams_in_chunks(file, text_col, chunk_size=20000, ner_enabled=True, spacy_batch_size=500):
    from collections import Counter
    bigram_counter = Counter()
    total_rows = 0
    for cdf in read_data_in_chunks(file, chunk_size=chunk_size):
        if text_col not in cdf.columns:
            continue
        cdf = cdf.dropna(subset=[text_col])
        texts = cdf[text_col].astype(str).tolist()
        for bg in generate_bigrams_from_texts(texts, ner_enabled=ner_enabled, batch_size=spacy_batch_size):
            bigram_counter[bg] += 1
        total_rows += len(cdf)
    return bigram_counter, total_rows

@st.cache_data
def compute_bigrams_all_at_once(df, text_col, ner_enabled=True, spacy_batch_size=500):
    from collections import Counter
    bigram_counter = Counter()
    texts = df[text_col].dropna().astype(str).tolist()
    for bg in generate_bigrams_from_texts(texts, ner_enabled=ner_enabled, batch_size=spacy_batch_size):
        bigram_counter[bg] += 1
    return bigram_counter, len(texts)

def auto_select_text_column(df: pd.DataFrame):
    priority_cols = ["Message","message","Text","text"]
    for c in priority_cols:
        if c in df.columns:
            return c
    str_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[c])]
    if not str_cols:
        return None
    best_col = max(str_cols, key=lambda c: df[c].dropna().apply(len).sum())
    return best_col

def create_bigram_graph(bigrams_count: dict, max_terms=50, min_freq=1):
    # Filter top bigrams
    items = [(bg, freq) for bg, freq in bigrams_count.items() if freq >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:max_terms]
    G = nx.DiGraph()
    for (w1, w2), freq in top:
        G.add_edge(w1, w2, weight=freq)
    return G

import adjustText

def reduce_label_overlap(ax, text_objs):
    # Increase expansions to reduce collisions
    adjustText.adjust_text(
        text_objs, ax=ax,
        force_text=3.0, force_points=3.0,
        expand_text=(2.0, 2.0), expand_points=(2.0, 2.0)
    )

# ---------------------------
# BFS + Balanced Spiral Layout with sfdp
# ---------------------------
def layout_components(G: nx.DiGraph, layout_prog="sfdp"):
    """
    For each connected component:
      - BFS from highest-betweenness node
      - Graphviz layout with 'sfdp' (force-directed) or fallback
      - Re-center, scale, random rotate
      - Spiral offset for subgraphs
    """
    UG = G.to_undirected()
    comps = list(nx.connected_components(UG))
    pos = {}

    base_radius = 400.0
    radius_step = 250.0
    angle_offset = 0.0

    for i, comp_set in enumerate(comps, start=1):
        sub_nodes = list(comp_set)
        sub_ug = UG.subgraph(sub_nodes).copy()
        sub_size = len(sub_nodes)

        bc = nx.betweenness_centrality(sub_ug)
        root = max(bc, key=bc.get) if bc else np.random.choice(sub_nodes)

        subD = nx.DiGraph()
        paths = nx.shortest_path(sub_ug, source=root)
        for target, path in paths.items():
            nx.add_path(subD, path)

        # Copy freq
        for (u,v) in subD.edges():
            if G.has_edge(u,v):
                subD[u][v]["weight"] = G[u][v]["weight"]
            elif G.has_edge(v,u):
                subD[u][v]["weight"] = G[v][u]["weight"]
            else:
                subD[u][v]["weight"] = 1

        # sfdp or fallback
        try:
            import pygraphviz
            try:
                sub_pos = nx.nx_agraph.graphviz_layout(subD, prog=layout_prog, root=root)
            except:
                sub_pos = nx.kamada_kawai_layout(subD)
        except ImportError:
            sub_pos = nx.kamada_kawai_layout(subD)

        xs = [sub_pos[n][0] for n in sub_pos]
        ys = [sub_pos[n][1] for n in sub_pos]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w = maxx - minx
        h = maxy - miny

        scale_factor = 1.0 + 0.07 * sub_size
        scale_factor = min(scale_factor, 4.0)

        import random
        for node in sub_pos:
            px, py = sub_pos[node]
            px -= (minx + w/2)
            py -= (miny + h/2)
            px *= scale_factor
            py *= scale_factor
            angle = math.radians(random.uniform(-15,15))
            rx = px*math.cos(angle) - py*math.sin(angle)
            ry = px*math.sin(angle) + py*math.cos(angle)
            sub_pos[node] = (rx, ry)

        sub_radius_bonus = sub_size * 10
        radius = base_radius + (i-1)*radius_step + sub_radius_bonus
        rand_angle = angle_offset + random.uniform(-0.5,0.5)
        offset_x = radius * math.cos(rand_angle)
        offset_y = radius * math.sin(rand_angle)
        angle_offset += math.radians(50)

        for node in sub_pos:
            px, py = sub_pos[node]
            pos[node] = (px + offset_x, py + offset_y)

    # fallback
    for node in G.nodes():
        if node not in pos:
            pos[node] = np.random.rand(2)*1000.0

    return pos

def build_bigram_figure(G: nx.DiGraph, layout_prog="sfdp", dpi=300, title="Bigram Network Graph"):
    if len(G)==0:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.axis("off")
        return fig, ax

    fig, ax = plt.subplots(figsize=(16,10), dpi=dpi)

    # layout BFS + spiral + sfdp
    pos = layout_components(G, layout_prog=layout_prog)

    degrees = dict(G.degree())
    node_sizes = [(degrees[n] + 2)**1.5 * 25 for n in G.nodes()]
    node_colors = [math.log10(degrees[n]+1) for n in G.nodes()]

    freq_vals = [G[u][v]["weight"] for (u,v) in G.edges()]
    mn, mx = min(freq_vals), max(freq_vals)
    rng = mx - mn if mx>mn else 1

    edge_widths = []
    edge_colors = []
    for (u,v) in G.edges():
        f = G[u][v]["weight"]
        w = 0.5 + 3.0*(f - mn)/rng
        edge_widths.append(w)
        edge_colors.append(plt.cm.magma((f - mn)/rng))

    # Draw nodes & edges
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.magma,
        alpha=0.9,
        linewidths=1.0,
        edgecolors="black"
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True, arrowstyle='-|>',
        arrowsize=25,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.85,
        connectionstyle='arc3,rad=0.3'
    )

    # Place labels
    text_objs = []
    for n,(x,y) in pos.items():
        lbl = ax.text(
            x, y+0.1,
            n,
            fontsize=14, fontweight='bold', color='black',
            ha='center', va='bottom'
        )
        text_objs.append(lbl)

    # Reduce label collisions
    reduce_label_overlap(ax, text_objs)

    # Color bars on opposite sides
    node_vals = list(node_colors)
    sm_nodes = plt.cm.ScalarMappable(cmap=plt.cm.magma,
        norm=plt.Normalize(vmin=min(node_vals), vmax=max(node_vals)))
    sm_nodes.set_array([])
    # Move node color bar to the LEFT
    cbar_nodes = fig.colorbar(sm_nodes, ax=ax, fraction=0.02, pad=0.01, location='left')
    cbar_nodes.set_label("Term Frequency (Log Scale)", rotation=90, labelpad=8)

    sm_edges = plt.cm.ScalarMappable(cmap=plt.cm.magma,
        norm=plt.Normalize(vmin=mn, vmax=mx))
    sm_edges.set_array([])
    # Move edge color bar to the RIGHT
    cbar_edges = fig.colorbar(sm_edges, ax=ax, fraction=0.02, pad=0.03, location='right')
    cbar_edges.set_label("N-gram Frequency", rotation=270, labelpad=8)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis("off")
    return fig, ax

def download_graph_as_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

def convert_excel_to_csv_in_memory(df: pd.DataFrame):
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

def scroll_to_button():
    if "has_scrolled" not in st.session_state:
        st.session_state["has_scrolled"] = True
        st.rerun()

# ---------------------------
# Main
# ---------------------------
def main():
    st.write("## Configuration")
    # Default parameter changes:
    # max_terms => default=45
    # min_freq => default=10
    max_terms = st.slider("How many terms do you want in your graph?", 5, 75, 45, step=5)
    min_freq = st.slider("How many times does a term need to be said to be included?", 1, 50, 10, step=1)

    ner_enabled = not st.checkbox("Disable NER (faster, but no entity recognition)", value=False)
    chunk_mode = st.checkbox("Large CSV? (Use chunk reading) [>30MB files]", value=False)

    layout_choice = st.radio(
        "Choose Visualization Style:",
        ("Radial Web (sfdp)", "Hierarchical Flow (dot)"),
        index=0
    )

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"], accept_multiple_files=False)

    if uploaded_file:
        global_df = None
        scroll_to_button()

        is_excel = uploaded_file.name.lower().endswith(".xlsx")
        if is_excel:
            st.info("Loading Excel → converting to CSV in memory…")
            df_excel = pd.read_excel(uploaded_file)
            if df_excel.empty:
                st.error("No data or file is empty!")
                return
            csv_like = convert_excel_to_csv_in_memory(df_excel)
        else:
            csv_like = uploaded_file

        if not chunk_mode:
            try:
                global_df = pd.read_csv(csv_like, encoding='utf-8', sep=None, engine='python')
                csv_like.seek(0)
                st.write("**Data Preview (First 5 Rows)**")
                st.dataframe(global_df.head(5))
            except Exception as e:
                st.error(f"Failed reading CSV: {e}. Try 'Large CSV' mode.")
                return

    st.write("---")
    generate_button = st.button("Generate Bigram Network")

    if uploaded_file and generate_button:
        if chunk_mode and not is_excel:
            st.write("**Extracting bigrams in chunks...**")
            csv_like.seek(0)
            try:
                first_chunk = next(read_data_in_chunks(csv_like, 1000))
            except StopIteration:
                st.error("File appears empty or invalid.")
                return
            col_candidate = auto_select_text_column(first_chunk)
            if col_candidate is None:
                st.error("No suitable text column found.")
                return
            possible_str = [c for c in first_chunk.columns if pd.api.types.is_string_dtype(first_chunk[c])]
            if col_candidate in possible_str and len(possible_str)>1:
                selected_col = st.selectbox("Select text column:", possible_str, index=possible_str.index(col_candidate))
            else:
                st.info(f"Auto-detected column: {col_candidate}")
                selected_col = col_candidate

            csv_like.seek(0)
            bigram_counts, total_rows = compute_bigrams_in_chunks(csv_like, selected_col, 20000, ner_enabled=ner_enabled)
            if not bigram_counts:
                st.warning("No bigrams found.")
                return
            st.success(f"Done! Processed ~{total_rows} rows.")
        else:
            if 'global_df' not in locals() or global_df is None:
                csv_like.seek(0)
                try:
                    global_df = pd.read_csv(csv_like, encoding='utf-8', sep=None, engine='python')
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")
                    return
                if global_df.empty:
                    st.error("No data or file is empty!")
                    return

            col_candidate = auto_select_text_column(global_df)
            if col_candidate is None:
                st.error("No suitable text column.")
                return
            possible_str = [c for c in global_df.columns if pd.api.types.is_string_dtype(global_df[c])]
            if col_candidate in possible_str and len(possible_str)>1:
                selected_col = st.selectbox("Select text column:", possible_str, index=possible_str.index(col_candidate))
            else:
                st.info(f"Auto-detected column: {col_candidate}")
                selected_col = col_candidate

            st.write("**Extracting bigrams...**")
            bigram_counts, total_rows = compute_bigrams_all_at_once(global_df, selected_col, ner_enabled=ner_enabled)
            if not bigram_counts:
                st.warning("No bigrams found.")
                return
            st.success(f"Done! Processed {total_rows} rows.")

        # Build bigram table
        bigram_df = pd.DataFrame(list(bigram_counts.items()), columns=["Bigram","Frequency"])
        bigram_df.sort_values("Frequency", ascending=False, inplace=True)

        with st.container():
            col1, col2 = st.columns([1,3])
            with col1:
                st.subheader("Top Bigrams (First 50)")
                st.dataframe(bigram_df.head(50))

            with col2:
                st.subheader("Bigram Network Graph")
                # Create graph
                G = create_bigram_graph(bigram_counts, max_terms=max_terms, min_freq=min_freq)
                if len(G) == 0:
                    st.warning("No edges meet your threshold.")
                    return

                if layout_choice == "Radial Web (sfdp)":
                    layout_prog = "sfdp"
                    final_title = "Bigram Network Graph (sfdp Radial)"
                else:
                    layout_prog = "dot"
                    final_title = "Bigram Network Graph (Hierarchical Flow)"

                fig, _ = build_bigram_figure(G, layout_prog=layout_prog, dpi=300, title=final_title)
                st.pyplot(fig, use_container_width=True)

        # Download CSV
        csv_data = bigram_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Bigrams (CSV)",
            data=csv_data,
            file_name="bigrams.csv",
            mime="text/csv"
        )

        # Download PNG
        png_data = download_graph_as_png(fig)
        if layout_choice == "Radial Web (sfdp)":
            st.download_button("Download Radial (PNG)", data=png_data, file_name="bigram_radial.png", mime="image/png")
        else:
            st.download_button("Download Hierarchical (PNG)", data=png_data, file_name="bigram_hierarchical.png", mime="image/png")


if __name__ == "__main__":
    main()
