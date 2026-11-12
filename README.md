# sing-langchain-rag-demo1

整体思路是：
    - 先实现可以对文档（pdf，md）进行切分，向量化和存储。
    - 先做一做非核心工作：
        - config.py文件
            - 一些常用的参数
            - 数据库信息

        - 测试数据库连接

        - 测试FastApi的运行

        - 先跑同RAG
            - 这次使用langChain编排RAG
            - pdf文件
            - md文件
            