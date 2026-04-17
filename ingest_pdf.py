import os
import sys
from pathlib import Path

# PDF 文件路径
PDF_FILE_PATH = r"D:\ai_test\智慧停车系统API文档.pdf"

# 向量数据库存储路径
DB_PATH = "./my_chroma_db"

# 集合名称
COLLECTION_NAME = "my_rag_collection"

from agent_components.chromadb_file import ReadersChromadb


def main():
    print("🚀 === 开始构建向量数据库 ===")

    try:
        # 2. 初始化数据库连接器
        print(f"🔗 正在连接数据库: {DB_PATH} ...")
        db_client = ReadersChromadb(
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME
        )

        # 3. 处理 PDF (读取 + 切分)
        print(f"📄 正在读取 PDF: {os.path.basename(PDF_FILE_PATH)}")
        documents = db_client.process_pdf_to_docs(PDF_FILE_PATH)

        if not documents:
            print("⚠️ 警告: PDF 解析后未获取到任何内容，可能是扫描版或加密文件。")
            return

        # 4. 存入向量数据库
        db_client.add_documents(documents)

        print("\n✅ === 数据库构建完成 ===")
        print(f"💡 提示: 请确保 Agent 启动时使用了相同的 DB_PATH 和 COLLECTION_NAME")

    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()