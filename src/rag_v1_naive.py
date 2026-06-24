# src/rag_v1_naive.py
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from openai import OpenAI

load_dotenv()

PG_CONN = os.getenv("DATABASE_URL")
COLLECTION = "diabetes_guideline_v1"   # 相当于"表/命名空间"

# ---------- 1. 加载 PDF ----------
def load_pdf(path: str) -> list[Document]:
    """
    使用 pypdf 直接读取 PDF，并转换为 LangChain 的 Document 对象。
    """
    reader = PdfReader(path)
    
    # 【附加优化】：处理 PDF 底层加密问题
    # 很多 PDF 默认带有空密码加密，这里尝试自动解密
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # 尝试使用空密码解密
        except Exception:
            print(f"⚠️ 警告：{path} 有密码保护且无法自动解密，请手动提供密码！")
            return []

    pages = []
    # 遍历每一页
    for i, page in enumerate(reader.pages):
        # 提取当前页的文本
        text = page.extract_text()
        
        # 构造 LangChain 的 Document 对象
        # 必须保留 source 和 page 元数据，这对下游的切片和检索溯源非常重要
        doc = Document(
            page_content=text,
            metadata={"source": path, "page": i}
        )
        pages.append(doc)
        
    print(f"[1] 加载了 {len(pages)} 页")
    return pages

# ---------- 2. 切片 ----------
def split(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"[2] 切成 {len(chunks)} 个 chunk")
    return chunks

# ---------- 3. Embedding + 写入 PGVector ----------
def build_vectorstore(chunks, rebuild: bool = False):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=PG_CONN,
        use_jsonb=True,
    )
    
    if rebuild:
        vectorstore.drop_tables()
        vectorstore.create_tables_if_not_exists()
        vectorstore.create_collection()
        vectorstore.add_documents(chunks)
        print(f"[3] 写入 {len(chunks)} 条向量到 PG")
    else:
        print(f"[3] 复用已有向量库")
    
    return vectorstore

# ---------- 4. 检索 + 生成 ----------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def ask(vectorstore, question: str, k: int = 4):
    docs = vectorstore.similarity_search_with_score(question, k=k)
    
    context_parts = []
    for i, (doc, score) in enumerate(docs):
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[片段{i+1} | 第{page}页 | 相似度{score:.3f}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""你是一名医疗辅助助手。请基于下面的指南内容回答问题。

【指南内容】
{context}

【问题】
{question}

【回答】"""
    
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    answer = resp.choices[0].message.content
    
    print("=" * 70)
    print(f"❓ 问题: {question}")
    print("-" * 70)
    print("📚 检索片段:")
    for i, (doc, score) in enumerate(docs):
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] p.{page} | score={score:.3f} | {preview}...")
    print("-" * 70)
    print(f"🤖 回答:\n{answer}")
    print("=" * 70)
    return answer

# ---------- 5. 主流程 ----------
if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    
    if rebuild:
        pages = load_pdf("./docs/diabetes_guideline.pdf")
        chunks = split(pages)
        vs = build_vectorstore(chunks, rebuild=True)
    else:
        vs = build_vectorstore([], rebuild=False)
    
    test_questions = [
        "2型糖尿病的诊断标准是什么？",
        "一个65岁男性,空腹血糖8.2,有高血压,该用什么降糖药？",
        "二甲双胍和恩格列净能一起用吗？",
        "我妈妈打胰岛素后头晕,该怎么办？",
        "糖尿病能根治吗？吃什么保健品好？",
    ]
    
    for q in test_questions:
        ask(vs, q)
        input("\n按回车继续下一题...\n")