import logging
from threading import Lock
from typing import Dict, List, Optional
from typing_extensions import TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from backend.rag.self_rag.self_rag_nodes import (
    GraphState,
    generate,
    grade_documents,
    transform_query,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)

logger = logging.getLogger(__name__)


def self_rag_workflow() -> StateGraph:
    # Initialize workflow
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query

    # Build graph
    workflow.add_edge(START, "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile
    app = workflow.compile()
    return app
