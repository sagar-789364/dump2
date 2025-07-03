import os
import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
import numpy as np

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Azure imports
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField
)
from pages.Usecase_1.pdf_image_embedd_azure_index import create_index
from pages.Usecase_1.sk_plugins_uc1 import (
    AzureConfig,
    DocumentRetrievalPlugin,
    ResponseGenerationPlugin,
    DocumentExportPlugin,
    ConversationMemoryPlugin
)

config = AzureConfig()
search_endpoint = config.search_endpoint
search_api_key = config.search_api_key
file_index_name = config.file_index_name
memory_index_name = config.memory_index_name

credential = AzureKeyCredential(search_api_key)
idx_client = SearchIndexClient(search_endpoint, credential)

# Use the robust index existence check/creation from pdf_image_embedd_azure_index.py
try:
    idx_client.get_index(file_index_name)
    print(f"Index '{file_index_name}' already exists.")
except ResourceNotFoundError:
    print(f"Index '{file_index_name}' not found. Creating...")
    create_index(search_endpoint, credential, file_index_name)
    print(f"Created {file_index_name}")


@dataclass
class QueryContext:
    author: List[str]
    scenario: str
    query: str
    session_id: str
    timestamp: datetime
    user: str = ""  # Add user field
    framework: str = ""
    accounting_standards: str = ""
    topics: str = ""

# Kernel Setup
async def setup_kernel(config: AzureConfig) -> Kernel:
    """Setup kernel with Azure OpenAI services"""
    kernel = Kernel()
    
    # Add Azure OpenAI chat completion service
    kernel.add_service(AzureChatCompletion(
        deployment_name=config.chat_deployment,
        endpoint=config.openai_endpoint,
        api_key=config.openai_api_key,
        api_version=config.openai_api_version
    ))
    
    # Add Azure OpenAI embedding service
    kernel.add_service(AzureTextEmbedding(
        deployment_name=config.embedding_deployment,
        endpoint=config.openai_endpoint,
        api_key=config.openai_api_key,
        api_version=config.openai_api_version
    ))
    return kernel


# Memory Manager
class RAGMemoryManager:
    """Manages different types of memory for the RAG system"""
    
    def __init__(self):
        # Semantic Memory: Facts about users, authors, preferences
        self.semantic_memory = {}
        
        # Episodic Memory: Past successful interactions
        self.episodic_memory = []
        
        # Procedural Memory: System prompt refinements
        self.procedural_memory = {
            "system_prompts": {},
            "feedback_patterns": []
        }
    
    def store_semantic_fact(self, key: str, value: Any, category: str = "general"):
        """Store semantic facts"""
        self.semantic_memory[key] = {
            'value': value,
            'category': category,
            'timestamp': datetime.now(),
            'access_count': 0
        }
    
    
    def store_episode(self, context: QueryContext, response: str, success_score: float):
        """Store successful interaction patterns"""
        episode = {
            'context': asdict(context),
            'response': response,
            'success_score': success_score,
            'timestamp': datetime.now()
        }
        self.episodic_memory.append(episode)
        
        # Keep only top episodes (memory consolidation)
        self.episodic_memory.sort(key=lambda x: x['success_score'], reverse=True)
        self.episodic_memory = self.episodic_memory[:50]  # Keep top 50
        


class SemanticRAGOrchestrator:
    def __init__(self, kernel: Kernel, memory_manager: Optional[RAGMemoryManager] = None):
        self.kernel = kernel
        self.memory_manager = memory_manager or RAGMemoryManager()

        self.chat_service = kernel.get_service(type=AzureChatCompletion)
        # Separate settings for new responses and reformatting
        self.execution_settings = AzureChatPromptExecutionSettings(
            max_tokens=2000,
            temperature=0.0
        )

        self.last_response_data = None
        
    async def process_accounting_query(
        self, 
        authors: List[str], 
        scenario: str, 
        query: str, 
        filename: str = None,
        user: str = "",
        framework: str = "",
        accounting_standards: str = "",
        topics: str = ""
    ) -> str:
        try:
            print("\n🔄 Starting Query Processing")
            memory_plugin = self.kernel.plugins["ConversationMemory"]
            document_plugin = self.kernel.plugins["DocumentRetrieval"]
            response_plugin = self.kernel.plugins["ResponseGeneration"]

            all_author_responses = []

            # Loop through each author separately
            for author in authors:
                print(f"\n--- Processing for author: {author} ---")
                similarity_args = KernelArguments(
                    settings=self.execution_settings,
                    function_choice_behavior=FunctionChoiceBehavior.Auto()
                )
                similarity_args["authors"] = [author]  # Use list for memory index
                similarity_args["scenario"] = scenario
                similarity_args["query"] = query
                similarity_args["user"] = user
                similarity_args["framework"] = framework
                similarity_args["accounting_standards"] = accounting_standards
                similarity_args["topics"] = topics

                # Check for similar conversations for this author
                similar_result = await memory_plugin["retrieve_similar_conversations"].invoke(
                    kernel=self.kernel,
                    arguments=similarity_args
                )
                similar_data = json.loads(str(similar_result))

                if similar_data.get("found", False):
                    print("\n♻️ Using existing response (Similarity found)")
                    similar_conv = similar_data["conversation"]
                    try:
                        history = ChatHistory()
                        system_message = f"""
    You are a helpful assistant that reformats accounting guidance responses.

    Previous Response to a Similar Query:
    {str(similar_conv['response'])}

    Please reformat this response for the following new query:
    Query: {query}
    Scenario: {scenario}

    Maintain the same structure but adjust the language and formatting:
        ## Guidance:  \n
        [Reformatted guidance]
        ## Sample Filings:  \n
        [Reformatted examples]
        ## Sources:  \n
        [Same sources, reformatted]

    Keep all technical information and examples but adjust the presentation to match the new query context.
    """
                        history.add_system_message(system_message)
                        history.add_user_message("Please reformat the response for the new query context.")

                        completion = await self.chat_service.get_chat_message_content(
                            chat_history=history,
                            settings=self.execution_settings,
                            kernel=self.kernel,
                            arguments=similarity_args
                        )
                        response = str(completion.content) if hasattr(completion, 'content') else str(completion)
                    except Exception as chat_error:
                        print(f"Warning: Chat completion failed: {chat_error}")
                        response = similar_conv['response']
                else:
                    print("\n🆕 Generating new response (No similar queries found)")
                    doc_args = KernelArguments(
                        settings=self.execution_settings,
                        function_choice_behavior=FunctionChoiceBehavior.Auto()
                    )
                    doc_args["author"] = author  # Use string for file index
                    doc_args["scenario"] = scenario
                    doc_args["query"] = query
                    doc_args["user"] = user
                    doc_args["framework"] = framework
                    doc_args["accounting_standards"] = accounting_standards
                    doc_args["topics"] = topics

                    documents_result = await document_plugin["retrieve_documents"].invoke(
                        kernel=self.kernel,
                        arguments=doc_args
                    )
                    response_args = KernelArguments(
                        settings=self.execution_settings,
                        function_choice_behavior=FunctionChoiceBehavior.Auto()
                    )
                    response_args["documents_json"] = str(documents_result)
                    response_args["scenario"] = scenario
                    response_args["query"] = query
                    response_args["user"] = user
                    response_args["framework"] = framework
                    response_args["accounting_standards"] = accounting_standards
                    response_args["topics"] = topics

                    response_obj = await response_plugin["generate_response"].invoke(
                        kernel=self.kernel,
                        arguments=response_args
                    )
                    response = str(response_obj)

                    # Store in memory only for new responses
                    context = QueryContext(
                        author=[author],
                        scenario=scenario,
                        query=query,
                        session_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        user=user,
                        framework=framework,
                        accounting_standards=accounting_standards,
                        topics=topics
                    )
                    self.memory_manager.store_semantic_fact("last_author", [author], category="author")
                    self.memory_manager.store_semantic_fact("last_scenario", scenario, category="scenario")
                    self.memory_manager.store_episode(context, response, success_score=1.0)

                    memory_args = KernelArguments(
                        settings=self.execution_settings,
                        function_choice_behavior=FunctionChoiceBehavior.Auto()
                    )
                    memory_args["authors"] = [author]  # Use list for memory index
                    memory_args["scenario"] = scenario
                    memory_args["query"] = query
                    memory_args["response"] = response
                    memory_args["user"] = user
                    memory_args["framework"] = framework
                    memory_args["accounting_standards"] = accounting_standards
                    memory_args["topics"] = topics

                    result = await memory_plugin["export_memory"].invoke(
                        kernel=self.kernel,
                        arguments=memory_args
                    )
                    print("✅ Response saved to knowledge base")

                # Prefix each response with author heading
                formatted_response = f"## {author}\n{response}"
                all_author_responses.append(formatted_response)

            # Combine all author responses
            final_response = "\n\n".join(all_author_responses)

            self.last_response_data = {
                'response': final_response,
                'scenario': scenario,
                'query': query,
                'author': author,
                'framework': framework,
                'accounting_standards': accounting_standards,
                'topics': topics
            }

            return final_response

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return f"Error: {str(e)}"


    async def generate_document(self, filename: str = None) -> bytes:
        """Generate document using the DocumentExportPlugin"""
        if not self.last_response_data:
            raise ValueError("No response data available. Please process a query first.")
        
        if not filename:
            filename = f"conversation_{uuid.uuid4().hex[:8]}.docx"
        
        try:
            export_plugin = self.kernel.plugins["DocumentExport"]
            export_args = KernelArguments(
                settings=self.execution_settings,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
            export_args["response"] = self.last_response_data['response']
            export_args["scenario"] = self.last_response_data['scenario']
            export_args["query"] = self.last_response_data['query']
            export_args["filename"] = filename
            # Add new fields to export_args
            export_args["framework"] = self.last_response_data.get('framework', "")
            export_args["accounting_standards"] = self.last_response_data.get('accounting_standards', "")
            export_args["topics"] = self.last_response_data.get('topics', "")
            result = await export_plugin["save_to_docx"].invoke(
                self.kernel,
                arguments=export_args
            )
            result_data = json.loads(str(result))
            if "error" in result_data:
                raise Exception(result_data["error"])
            import base64
            doc_bytes = base64.b64decode(result_data["document_data"])
            return doc_bytes
        except Exception as e:
            print(f"Document generation error: {e}")
            raise e

async def generate_guidance_response(authors: List[str], scenario: str, query: str, user: str = "", framework: str = "", accounting_standards: str = "", topics: str = "", config=AzureConfig()):
    # Setup kernel
    kernel = await setup_kernel(config)
    
    # Initialize memory manager
    memory_manager = RAGMemoryManager()
    
    # Initialize plugins
    retrieval_plugin = DocumentRetrievalPlugin(config)
    response_plugin = ResponseGenerationPlugin()
    export_plugin = DocumentExportPlugin()
    conversation_memory_plugin = ConversationMemoryPlugin(config)  # New combined plugin

    # Add plugins to kernel
    kernel.add_plugin(retrieval_plugin, plugin_name="DocumentRetrieval")
    kernel.add_plugin(response_plugin, plugin_name="ResponseGeneration")
    kernel.add_plugin(export_plugin, plugin_name="DocumentExport")
    kernel.add_plugin(conversation_memory_plugin, plugin_name="ConversationMemory")  # Add combined plugin

    # Initialize orchestrator
    chat_orchestrator = SemanticRAGOrchestrator(kernel, memory_manager)
    filename = f"{uuid.uuid4()}.docx"
    response = await chat_orchestrator.process_accounting_query(authors, scenario, query, filename, user, framework, accounting_standards, topics)
    return response, chat_orchestrator

def generate_guidance_response_sync(authors: List[str], scenario: str, query: str, user: str = "", framework: str = "", accounting_standards: str = "", topics: str = ""):
    """Synchronous wrapper that returns both response and orchestrator"""
    return asyncio.run(generate_guidance_response(authors, scenario, query, user, framework, accounting_standards, topics))

def generate_document_sync(orchestrator: SemanticRAGOrchestrator, filename: str = None) -> bytes:
    """Synchronous wrapper for document generation"""
    return asyncio.run(orchestrator.generate_document(filename))

def get_author_options_from_index(
    config: AzureConfig = AzureConfig(),
    author_field: str = "author"
) -> list:
    """
    Retrieve unique author options from the Azure Cognitive Search index.

    Args:
        search_endpoint (str): The endpoint of the Azure Cognitive Search service.
        search_key (str): The API key for the search service.
        file_index_name (str): The name of the search index.
        author_field (str): The field name in the index that contains author names.

    Returns:
        list: A list of unique author names.
    """
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.file_index_name,
        credential=AzureKeyCredential(config.search_api_key)
    )

    # Use facets to get unique author values
    results = client.search(
        search_text="*",
        facets=[author_field],
        top=0  # We only want facets, not actual documents
    )
    author_options = []
    facet_data = results.get_facets().get(author_field, [])
    # Debug print
    print(f"[DEBUG] Author facet data: {facet_data}")
    for facet in facet_data:
        author_options.append(facet['value'])
    return author_options

def get_framework_options_from_index(
    config: AzureConfig = AzureConfig(),
    framework_field: str = "framework"
) -> list:
    """
    Retrieve unique framework options from the Azure Cognitive Search index.

    Args:
        search_endpoint (str): The endpoint of the Azure Cognitive Search service.
        search_key (str): The API key for the search service.
        file_index_name (str): The name of the search index.
        framework_field (str): The field name in the index that contains framework names.

    Returns:
        list: A list of unique framework names.
    """
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.file_index_name,
        credential=AzureKeyCredential(config.search_api_key)
    )

    # Use facets to get unique framework values
    results = client.search(
        search_text="*",
        facets=[framework_field],
        top=0  # We only want facets, not actual documents
    )
    framework_options = []
    facet_data = results.get_facets().get(framework_field, [])
    print(f"[DEBUG] Framework facet data: {facet_data}")
    for facet in facet_data:
        framework_options.append(facet['value'])
    return framework_options

def get_accounting_standards_options_from_index(
    config: AzureConfig = AzureConfig(),
    accounting_standards_field: str = "accounting_standards"
) -> list:
    """
    Retrieve unique accounting standards options from the Azure Cognitive Search index.

    Args:
        search_endpoint (str): The endpoint of the Azure Cognitive Search service.
        search_key (str): The API key for the search service.
        file_index_name (str): The name of the search index.
        accounting_standards_field (str): The field name in the index that contains accounting standards names.

    Returns:
        list: A list of unique accounting standards names.
    """
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.file_index_name,
        credential=AzureKeyCredential(config.search_api_key)
    )

    # Use facets to get unique accounting standards values
    results = client.search(
        search_text="*",
        facets=[accounting_standards_field],
        top=0  # We only want facets, not actual documents
    )
    accounting_standards_options = []
    facet_data = results.get_facets().get(accounting_standards_field, [])
    print(f"[DEBUG] Accounting Standards facet data: {facet_data}")
    for facet in facet_data:
        accounting_standards_options.append(facet['value'])
    return accounting_standards_options

def get_topics_options_from_index(
    config: AzureConfig = AzureConfig(),
    topics_field: str = "topics"
) -> list:
    """
    Retrieve unique topics options from the Azure Cognitive Search index.

    Args:
        search_endpoint (str): The endpoint of the Azure Cognitive Search service.
        search_key (str): The API key for the search service.
        file_index_name (str): The name of the search index.
        topics_field (str): The field name in the index that contains topics.

    Returns:
        list: A list of unique topics.
    """
    client = SearchClient(
        endpoint=config.search_endpoint,
        index_name=config.file_index_name,
        credential=AzureKeyCredential(config.search_api_key)
    )

    # Use facets to get unique topics values
    results = client.search(
        search_text="*",
        facets=[topics_field],
        top=0  # We only want facets, not actual documents
    )
    topics_options = []
    facet_data = results.get_facets().get(topics_field, [])
    print(f"[DEBUG] Topics facet data: {facet_data}")
    for facet in facet_data:
        topics_options.append(facet['value'])
    return topics_options

