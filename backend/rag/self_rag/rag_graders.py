import logging
from threading import Lock
from typing import Dict, List, Optional, Tuple, Callable
from typing_extensions import TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        ..., description="Documents are relevant to the question, 'yes' or 'no'."
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucinations present in generated response."""

    binary_score: str = Field(
        ..., description="Answer is grounded in the facts, 'yes' or 'no'."
    )

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        ..., description="Answer addresses the question, 'yes' or 'no'."
    )


def create_graders(llm: BaseChatModel, grader_name: str) -> Tuple[Callable, Callable, Callable]:
    """Create graders for relevance, hallucinations, and answer."""


    grader_system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader = grade_prompt | structured_llm_grader

    hallucinations_system_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucinations_system_prompt),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

    relevance_system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
    answer_grader = answer_prompt | structured_llm_answer_grader


    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
        Return only one question."""
    
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    if grader_name == "retrieval":
        return retrieval_grader
    elif grader_name == "hallucination":
        return hallucination_grader
    elif grader_name == "answer":
        return answer_grader    
    elif grader_name == "question_rewriter":
        return question_rewriter
    else:
        return None
