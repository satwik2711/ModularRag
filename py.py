import json

notebook_content ={
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Advanced RAG Retrieval Strategies: Flow and Modular\n",
    "\n",
    "This notebook demonstrates the implementation of modular RAG and RAG flow for advanced retrieval strategies."
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Install required packages\n",
    "!pip install llama-index openai python-dotenv cohere langchain pyvis ragas datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "from llama_index.llms.openai import OpenAI\n",
    "from llama_index.embeddings.openai import OpenAIEmbedding\n",
    "from llama_index.core import (\n",
    "    Settings,\n",
    "    SimpleDirectoryReader,\n",
    "    StorageContext,\n",
    "    VectorStoreIndex,\n",
    "    load_index_from_storage,\n",
    ")\n",
    "from llama_index.core.node_parser import SentenceWindowNodeParser\n",
    "from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor\n",
    "from llama_index.core.query_pipeline import QueryPipeline, InputComponent, CustomQueryComponent\n",
    "from llama_index.core.response_synthesizers.tree_summarize import TreeSummarize\n",
    "from llama_index.postprocessor.cohere_rerank import CohereRerank\n",
    "from typing import Dict, Any\n",
    "from llama_index.core.indices.query.query_transform import HyDEQueryTransform\n",
    "from pyvis.network import Network\n",
    "from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall\n",
    "from ragas import evaluate\n",
    "from datasets import Dataset\n",
    "\n",
    "# Load environment variables\n",
    "load_dotenv()\n",
    "\n",
    "# Set up API keys\n",
    "os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')\n",
    "os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Index documents\n",
    "documents = SimpleDirectoryReader(\"./data\").load_data()\n",
    "node_parser = SentenceWindowNodeParser.from_defaults(\n",
    "    window_size=3,\n",
    "    window_metadata_key=\"window\",\n",
    "    original_text_metadata_key=\"original_text\",\n",
    ")\n",
    "llm = OpenAI(model=\"gpt-3.5-turbo\")\n",
    "embed_model = OpenAIEmbedding(model=\"text-embedding-3-small\")\n",
    "Settings.llm = llm\n",
    "Settings.embed_model = embed_model\n",
    "Settings.node_parser = node_parser\n",
    "\n",
    "if not os.path.exists(\"storage\"):\n",
    "    index = VectorStoreIndex.from_documents(documents)\n",
    "    index.set_index_id(\"avengers\")\n",
    "    index.storage_context.persist(\"./storage\")\n",
    "else:\n",
    "    store_context = StorageContext.from_defaults(persist_dir=\"./storage\")\n",
    "    index = load_index_from_storage(\n",
    "        storage_context=store_context, index_id=\"avengers\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Define custom components\n",
    "class HydeComponent(CustomQueryComponent):\n",
    "    \"\"\"HyDE query rewrite component.\"\"\"\n",
    "    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:\n",
    "        assert \"input\" in input, \"input is required\"\n",
    "        return input\n",
    "\n",
    "    @property\n",
    "    def _input_keys(self) -> set:\n",
    "        return {\"input\"}\n",
    "\n",
    "    @property\n",
    "    def _output_keys(self) -> set:\n",
    "        return {\"output\"}\n",
    "\n",
    "    def _run_component(self, **kwargs) -> Dict[str, Any]:\n",
    "        hyde = HyDEQueryTransform(include_original=True)\n",
    "        query_bundle = hyde(kwargs[\"input\"])\n",
    "        return {\"output\": query_bundle.embedding_strs[0]}\n",
    "\n",
    "class RagasComponent(CustomQueryComponent):\n",
    "    \"\"\"Ragas evaluation component.\"\"\"\n",
    "    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:\n",
    "        return input\n",
    "    \n",
    "    @property\n",
    "    def _input_keys(self) -> set:\n",
    "        return {\"question\", \"nodes\", \"answer\", \"ground_truth\"}\n",
    "\n",
    "    @property\n",
    "    def _output_keys(self) -> set:\n",
    "        return {\"answer\", \"source_nodes\", \"evaluation\"}\n",
    "\n",
    "    def _run_component(self, **kwargs) -> Dict[str, Any]:\n",
    "        question, ground_truth, nodes, answer = kwargs.values()\n",
    "        data = {\n",
    "            \"question\": [question],\n",
    "            \"contexts\": [[n.get_content() for n in nodes]],\n",
    "            \"answer\": [str(answer)],\n",
    "            \"ground_truth\": [ground_truth],\n",
    "        }\n",
    "        dataset = Dataset.from_dict(data)\n",
    "        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]\n",
    "        evaluation = evaluate(dataset, metrics)\n",
    "        return {\"answer\": str(answer), \"source_nodes\": nodes, \"evaluation\": evaluation}"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Set up query pipeline\n",
    "retriever = index.as_retriever()\n",
    "query_rewriter = HydeComponent()\n",
    "reranker = CohereRerank()\n",
    "meta_replacer = MetadataReplacementPostProcessor(target_metadata_key=\"window\")\n",
    "evaluator = RagasComponent()\n",
    "\n",
    "p = QueryPipeline(verbose=True)\n",
    "p.add_modules(\n",
    "    {\n",
    "        \"input\": InputComponent(),\n",
    "        \"query_rewriter\": query_rewriter,\n",
    "        \"retriever\": retriever,\n",
    "        \"meta_replacer\": meta_replacer,\n",
    "        \"reranker\": reranker,\n",
    "        \"output\": TreeSummarize(),\n",
    "        \"evaluator\": evaluator,\n",
    "    }\n",
    ")\n",
    "\n",
    "p.add_link(\"input\", \"query_rewriter\", src_key=\"input\")\n",
    "p.add_link(\"query_rewriter\", \"retriever\")\n",
    "p.add_link(\"retriever\", \"meta_replacer\")\n",
    "p.add_link(\"input\", \"reranker\", src_key=\"input\", dest_key=\"query_str\")\n",
    "p.add_link(\"meta_replacer\", \"reranker\", dest_key=\"nodes\")\n",
    "p.add_link(\"input\", \"output\", src_key=\"input\", dest_key=\"query_str\")\n",
    "p.add_link(\"reranker\", \"output\", dest_key=\"nodes\")\n",
    "p.add_link(\"input\", \"evaluator\", src_key=\"input\", dest_key=\"question\")\n",
    "p.add_link(\"input\", \"evaluator\", src_key=\"ground_truth\", dest_key=\"ground_truth\")\n",
    "p.add_link(\"reranker\", \"evaluator\", dest_key=\"nodes\")\n",
    "p.add_link(\"output\", \"evaluator\", dest_key=\"answer\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Visualize query pipeline\n",
    "net = Network(notebook=True, cdn_resources=\"in_line\", directed=True)\n",
    "net.from_nx(p.clean_dag)\n",
    "net.write_html(\"output/pipeline_dag.html\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Run query pipeline\n",
    "question = \"Which two members of the Avengers created Ultron?\"\n",
    "ground_truth = \"Tony Stark (Iron Man) and Bruce Banner (The Hulk).\"\n",
    "output = p.run(input=question, ground_truth=ground_truth)\n",
    "print(f\"Answer: {output['answer']}\")\n",
    "print(f\"Evaluation: {output['evaluation']}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_type": "null",
   "metadata": {},
   "source": [
    "# Print intermediate results (optional)\n",
    "output, intermediates = p.run_with_intermediates(input=question, ground_truth=ground_truth)\n",
    "\n",
    "print(\"Retriever output:\")\n",
    "for node in intermediates[\"retriever\"].outputs[\"output\"]:\n",
    "    print(f\"Node: {node.text}\\n\")\n",
    "\n",
    "print(\"Meta replacer output:\")\n",
    "for node in intermediates[\"meta_replacer\"].outputs[\"nodes\"]:\n",
    "    print(f\"Node: {node.text}\\n\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('rag_retrieval.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)