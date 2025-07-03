import fitz  # PyMuPDF for PDF→image
from datetime import datetime, timezone
from pypdf import PdfReader
import base64
import uuid
import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
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
from azure.core.exceptions import ResourceNotFoundError

from dotenv import load_dotenv
# Load the .env file
load_dotenv()

# Azure AI Search Configuration and data models
search_endpoint = os.getenv("search_endpoint")
search_api_key = os.getenv("search_api_key")
file_index_name = os.getenv("file_index_name")
memory_index_name = os.getenv("memory_index_name")

# Azure OpenAI Configuration
openai_api_version = os.getenv("openai_api_version")
openai_api_key = os.getenv("openai_api_key")
vision_model_base_url = os.getenv("vision_model_base_url")
embedding_model_base_url = os.getenv("embedding_model_base_url")

vision_deployment = "gpt-4o-mini"
embedding_deployment = "text-embedding-3-large"

vision_client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    base_url=vision_model_base_url
)

embedding_client = AzureOpenAI(
    api_key=openai_api_key,
    api_version=openai_api_version,
    base_url=embedding_model_base_url
)

# Create the index in azure
def create_index(search_endpoint, credential, index_name):
    from azure.search.documents.indexes.models import SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields, SemanticField
    client = SearchIndexClient(search_endpoint, credential)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True, facetable=True),
        SimpleField(name="document_title", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="upload_date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True, facetable=True),
        SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="framework", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="accounting_standards", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="topics", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="author", type=SearchFieldDataType.String, filterable=True, facetable=True),  # changed
        SimpleField(name="language", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="my-vector-config"
        )
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="my-algorithms-config")
        ]
    )

    # Add semantic configuration
    semantic_config = SemanticConfiguration(
        name="guidance-files-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="document_title"),
            keywords_fields=[
                SemanticField(field_name="framework"),
                SemanticField(field_name="accounting_standards"),
                SemanticField(field_name="topics"),
                SemanticField(field_name="author")  # changed
            ],
            content_fields=[
                SemanticField(field_name="text")
            ]
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
    return client, index

credential = AzureKeyCredential(search_api_key)

idx_client = SearchIndexClient(search_endpoint, credential)
# Robust check for index existence and creation
index_exists = False
try:
    idx_client.get_index(file_index_name)
    print(f"Index '{file_index_name}' already exists.")
    index_exists = True
except ResourceNotFoundError:
    print(f"Index '{file_index_name}' not found. Creating...")
    _, idx = create_index(search_endpoint, credential, file_index_name)
    idx_client.create_index(idx)
    print(f"Index '{file_index_name}' created successfully.")
except Exception as e:
    print(f"Error checking/creating index: {e}")
    raise

# Only create SearchClient after index is confirmed to exist
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=file_index_name,  # Use 'guidance' for file uploads
    credential=AzureKeyCredential(search_api_key)
)

# Function convert pages of pdf to images (pdfPath -> images) 
def pdfPath_to_images(pdf_path):
    reader = PdfReader(pdf_path)
    no_of_pages = len(reader.pages)
    first_page, last_page = 1, no_of_pages

    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(1, 1)
    images = [
        (pno + 1, doc[pno].get_pixmap(matrix=matrix).tobytes("png"))
        for pno in range(first_page - 1, last_page)
    ]

    return images

# Function to chunk text with overlap
def chunk_text(text, chunk_size=1024, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to prepare the embeddings of the text
def get_embedding(text, model, client=embedding_client):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Function to process the images using the vision model, extract text, convert to indexes and upload to azure AI search
def process_images_to_indexes_and_upload(pdf_path, output_file, vision_client=vision_client, vision_deployment=vision_deployment, 
                                         embedding_deployment=embedding_deployment, search_client=search_client,
                                         framework=None, accounting_standards=None, topics=None, author=None, progress_callback=None):  # changed param
    images = pdfPath_to_images(pdf_path)
    total_pages = len(images)
    import os
    document_title = os.path.splitext(os.path.basename(pdf_path))[0]
    source_file = document_title
    system_prompt = (
        '''You are an advanced OCR assistant that can perform the complex operations on any image as per the instructions. You are given images that may have text, images, tables, flow diagrams, other visual representations combined or individual. Your task is to extract all text and visual content from the image following the below guidelines:
        - If there is text, just extract it as it is. Maintain the consistency of the format, structure and style. (Don't interpret it unless it is asked to do so) 
        - If there are tables, extract them in correct tabular format and in correct order. 
        - If there are flow charts, extract their node/edge structure and describe them in detail and interpret their meaning. 
        - If there are images, extract any embedded text, describe the image, and analyze its content. 
        - If there are any other visual elements, describe what they represent and interpret them. 
        Do not skip any content—text, tables, images, flow charts, or other visuals. Maintain the order of the elements (text, images, tables, flow charts, etc) as per the image.'''
        )
    batch_docs = []
    batch_size = 5
    with open(output_file, "w", encoding="utf-8") as out_f:
        for page_idx, (page_num, img_bytes) in enumerate(images):
            if progress_callback:
                progress_callback(page_idx, total_pages)
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            resp = vision_client.chat.completions.create(
                model=vision_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": data_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            extracted = resp.choices[0].message.content
            out_f.write(f"\n--- Page {page_num} ---\n")
            out_f.write(extracted + "\n")
            # Chunk and prepare for upload
            chunks = chunk_text(extracted)
            if not chunks:
                print(f"No chunks generated for page {page_num}")
                continue
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk, embedding_deployment)
                batch_docs.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "page": page_num,
                    "document_title": document_title,
                    "author": author,  # changed
                    "source_file": source_file,
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                    "chunk_index": i,
                    "content_type": "text",
                    "language": "en",
                    "embedding": embedding,
                    "framework": framework,
                    "accounting_standards": accounting_standards,
                    "topics": topics
                })
            # Upload in batches of 5 pages
            if (page_idx + 1) % batch_size == 0 and batch_docs:
                try:
                    search_client.upload_documents(documents=batch_docs)
                except Exception as e:
                    print(f"Error uploading batch ending at page {page_num}: {e}")
                batch_docs = []
        # Upload any remaining docs
        if batch_docs:
            try:
                search_client.upload_documents(documents=batch_docs)
            except Exception as e:
                print(f"Error uploading final batch: {e}")
    print(f"Indexed {total_pages} pages and wrote OCR text to '{output_file}' of {document_title}.")
