from rag import build_index


if __name__ == "__main__":
    print(f"已建立索引，共切分 {build_index()} 个文本片段。")
