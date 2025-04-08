import json
import os
import time
import numpy as np
import pandas as pd
import faiss
import torch
import gradio as gr
from scholarly import scholarly
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# === 1. Load SPECTER2 with proximity adapter ===
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
model.eval()

# === 2. Get papers from Google Scholar ===
def get_org_papers(gs_profile_id, max_papers=20):
    author = scholarly.search_author_id(gs_profile_id)
    author_filled = scholarly.fill(author, sections=["publications"])
    paper_list = []
    for pub in author_filled["publications"][:max_papers]:
        try:
            pub_filled = scholarly.fill(pub)
            paper_list.append({
                "title": pub_filled.get("bib", {}).get("title", ""),
                "abstract": pub_filled.get("bib", {}).get("abstract", ""),
                "year": pub_filled.get("bib", {}).get("pub_year", ""),
                "venue": pub_filled.get("bib", {}).get("venue", ""),
                "authors": pub_filled.get("bib", {}).get("author", ""),
            })
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error loading pub: {e}")
            continue
    return paper_list

# === 3. Embed papers with SPECTER2 ===
def embed_papers(papers):
    texts = [p["title"] + tokenizer.sep_token + p.get("abstract", "") for p in papers]
    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# === 4. Build FAISS index ===
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# === 5. Save and Load Index ===
def save_index_and_data(embeddings, index, papers, dir_path="index_data"):
    os.makedirs(dir_path, exist_ok=True)
    faiss.write_index(index, os.path.join(dir_path, "index.faiss"))
    np.save(os.path.join(dir_path, "embeddings.npy"), embeddings)
    with open(os.path.join(dir_path, "metadata.json"), "w") as f:
        json.dump(papers, f)

def load_index_and_data(dir_path="index_data"):
    index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
    embeddings = np.load(os.path.join(dir_path, "embeddings.npy"))
    with open(os.path.join(dir_path, "metadata.json"), "r") as f:
        papers = json.load(f)
    return papers, embeddings, index

# === 6. Compute pairwise similarity table ===
def compute_pairwise_similarity_table(papers, embeddings, top_k=100):
    sims = cosine_similarity(embeddings)
    results = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            results.append({
                "paper_A_title": papers[i]["title"],
                "paper_B_title": papers[j]["title"],
                "paper_A_authors": papers[i].get("authors", ""),
                "paper_B_authors": papers[j].get("authors", ""),
                "similarity": round(sims[i, j], 4)
            })
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    df = pd.DataFrame(results[:top_k])
    df.to_csv("pairwise_similarity_top100.csv", index=False)
    return df

# === 7. Visualizations ===
def build_similarity_graph(pairwise_df, threshold=0.85):
    G = nx.Graph()
    for _, row in pairwise_df.iterrows():
        if row["similarity"] >= threshold:
            G.add_edge(row["paper_A_title"], row["paper_B_title"], weight=row["similarity"])
    return G

def draw_similarity_graph(G, figsize=(12, 8)):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray")
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()})
    plt.title("Semantic Similarity Graph")
    plt.show()

def build_interactive_graph(pairwise_df, threshold=0.85, output_file="similarity_graph.html"):
    net = Network(height="750px", width="100%", notebook=False)
    for _, row in pairwise_df.iterrows():
        if row["similarity"] >= threshold:
            a, b = row["paper_A_title"], row["paper_B_title"]
            net.add_node(a, title=row["paper_A_authors"], label=a)
            net.add_node(b, title=row["paper_B_authors"], label=b)
            net.add_edge(a, b, value=row["similarity"])
    net.show_buttons(filter_=["physics"])
    net.show(output_file)

# === 8. Gradio UI ===
def launch_gradio(papers, index, embeddings):
    def recommend(title, abstract, top_k):
        text = title + tokenizer.sep_token + abstract
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True,
                           return_token_type_ids=False, max_length=512)
        with torch.no_grad():
            output = model(**inputs)
        query_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
        faiss.normalize_L2(query_embedding)
        scores, indices = index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            paper = papers[idx]
            results.append(f"🔹 **{paper['title']}** ({paper['venue']}, {paper['year']})\nScore: {score:.3f}")
        return "\n\n".join(results)

    gr.Interface(
        fn=recommend,
        inputs=[
            gr.Textbox(label="Title"),
            gr.Textbox(label="Abstract"),
            gr.Slider(minimum=1, maximum=10, step=1, label="Top K Results", value=5)
        ],
        outputs=gr.Markdown(),
        title="📚 Citation Recommender",
        description="Find semantically similar papers from your Google Scholar corpus."
    ).launch()

# === 9. Run it all ===
google_scholar_id = "Zid_jw4AAAAJ"
try:
    papers, embeddings, index = load_index_and_data()
    print("✅ Loaded cached index and papers.")
except:
    print("🚀 Building index from scratch...")
    papers = get_org_papers(google_scholar_id, max_papers=30)
    embeddings = embed_papers(papers)
    index = build_index(embeddings)
    save_index_and_data(embeddings, index, papers)

pairwise_df = compute_pairwise_similarity_table(papers, embeddings, top_k=100)
G = build_similarity_graph(pairwise_df, threshold=0.88)
draw_similarity_graph(G)
build_interactive_graph(pairwise_df, threshold=0.88)

launch_gradio(papers, index, embeddings)
