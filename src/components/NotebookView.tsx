import { Download, Play, Lightbulb, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PipelineType } from "./ChatInterface";

interface NotebookCell {
  type: "markdown" | "code";
  content: string;
}

interface ExplainabilityData {
  bullets: string[];
  citation: string;
}

const notebookExplanations: Record<PipelineType, ExplainabilityData> = {
  rag: {
    bullets: [
      "**Mistral-7B selected** as the base model due to its strong performance on retrieval-augmented tasks and efficient inference for sub-2-second response times.",
      "**Chunk size of 512 tokens** with 50-token overlap chosen to balance context preservation with retrieval precision for long troubleshooting guides.",
      "**Elasticsearch integration** configured with document filtering (`product_docs`, `support_playbooks` tags) and top-3 passage retrieval to enable source-linked answers for agent verification."
    ],
    citation: "Recommended based on Red Hat OpenShift AI documentation, section \"Production RAG Pipelines\" (fictitious)."
  },
  finetuning: {
    bullets: [
      "**LoRA-based fine-tuning** selected to minimize GPU cost while achieving effective adaptation on the 7B-parameter model for email classification.",
      "**InstructLab synthetic data generation** with PII redaction and class balancing to address bias in historical email datasets.",
      "**Macro-F1 threshold of 0.85** set as the model registration gate to ensure reliable classification across all categories (auto, property, health, fraud_review)."
    ],
    citation: "Derived from Red Hat AI validated demos – Insurance Document Classification (fictitious)."
  },
  synthetic: {
    bullets: [
      "**K-anonymity enforcement** on sensitive CDR fields ensures synthetic data cannot be re-identified while preserving analytical utility.",
      "**1 million row target** with preserved churn-label correlations enables statistically valid model training without production PII access.",
      "**Automated utility validation** via baseline AUC comparison between real and synthetic data ensures the generated dataset maintains predictive power."
    ],
    citation: "Recommended based on Red Hat AI best practices and guides, section \"Privacy-Preserving Synthetic Data\" (fictitious)."
  },
  llmserving: {
    bullets: [
      "**llm-d disaggregated architecture** selected to separate prefill and decode phases, maximizing GPU utilization and enabling independent scaling for each workload type.",
      "**Tensor parallelism across 4 GPUs** per node distributes Llama-3-70B's 70B parameters efficiently, with FP16 precision balancing quality and memory footprint.",
      "**Prefix caching with 32GB allocation** reuses KV-cache across requests sharing common prompt prefixes, reducing time-to-first-token by up to 60% for enterprise chatbot patterns."
    ],
    citation: "Derived from Red Hat AI validated demos – Enterprise LLM Serving with llm-d (fictitious)."
  },
  agentic: {
    bullets: [
      "**Modular tool-based architecture** enables independent development and testing of each agent capability (triage, labeling, review suggestion, dependency checks) with clear interfaces.",
      "**GitHub webhook integration** with event routing ensures real-time response to repository events while the cron scheduler handles batch operations for stale issue cleanup.",
      "**OpenShift Secrets management** with automatic rotation secures GitHub tokens and API credentials, meeting enterprise compliance requirements for credential handling."
    ],
    citation: "Derived from Red Hat AI validated demos – Agentic Workflow Starter Kits (fictitious)."
  }
};

const notebookConfigs: Record<PipelineType, NotebookCell[]> = {
  rag: [
    {
      type: "markdown",
      content: "# RAG Optimization Pipeline\nWatsonx AutoAI-style RAG pipeline with document processing, chunking, embedding, and pattern evaluation",
    },
    {
      type: "code",
      content: `import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import Embeddings`,
    },
    {
      type: "markdown",
      content: "## Step 1: Document Loading and Parsing\nLoad grounding documents from various sources (PDF, Word, text files)",
    },
    {
      type: "code",
      content: `from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_documents(file_paths):
    """Load documents from various file types"""
    documents = []
    
    for path in file_paths:
        if path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        elif path.endswith('.txt'):
            loader = TextLoader(path)
        elif path.endswith('.docx'):
            loader = Docx2txtLoader(path)
        
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} documents")
    return documents

docs = load_documents(['doc1.pdf', 'doc2.txt', 'doc3.docx'])`,
    },
    {
      type: "markdown",
      content: "## Step 2: Document Chunking Strategy\nSplit documents into optimal chunk sizes with configurable overlap",
    },
    {
      type: "code",
      content: `# Configure recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=["\\n\\n", "\\n", " ", ""]
)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks with avg size {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")`,
    },
    {
      type: "markdown",
      content: "## Step 3: Embedding Generation with Watsonx\nConvert chunks to vector embeddings using slate-30m-english-rtrvr model",
    },
    {
      type: "code",
      content: `# Initialize watsonx.ai embedding model
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "YOUR_API_KEY"
}

embeddings = Embeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    credentials=credentials,
    project_id="YOUR_PROJECT_ID"
)

# Generate embeddings for chunks
print("Generating embeddings...")
embedded_chunks = embeddings.embed_documents([c.page_content for c in chunks])
print(f"Generated {len(embedded_chunks)} embeddings of dimension {len(embedded_chunks[0])}")`,
    },
    {
      type: "markdown",
      content: "## Step 4: Vector Store Setup\nStore vectorized content in Milvus database for efficient retrieval",
    },
    {
      type: "code",
      content: `from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Create vector store
vectorstore = Milvus.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_knowledge_base",
    connection_args={"host": "localhost", "port": "19530"}
)

print("Vector store created successfully!")`,
    },
    {
      type: "markdown",
      content: "## Step 5: Retrieval Testing\nTest retrieval accuracy with sample queries and ground truth",
    },
    {
      type: "code",
      content: `# Test retrieval with sample queries
test_queries = [
    "What is the main purpose of the system?",
    "How does the authentication process work?",
    "What are the performance benchmarks?"
]

retrieval_results = []
for query in test_queries:
    docs = vectorstore.similarity_search(query, k=5)
    retrieval_results.append({
        'query': query,
        'num_results': len(docs),
        'top_result': docs[0].page_content[:200] if docs else None
    })

print(f"Tested {len(test_queries)} queries with avg {sum(r['num_results'] for r in retrieval_results) / len(test_queries):.1f} results")`,
    },
    {
      type: "markdown",
      content: "## Step 6: RAG Pattern Optimization with AutoAI\nRun AutoAI RAG experiment to evaluate and rank different RAG patterns",
    },
    {
      type: "code",
      content: `from ibm_watsonx_ai.experiment import AutoAI

# Configure RAG optimizer
rag_optimizer = AutoAI(
    credentials=credentials,
    project_id="YOUR_PROJECT_ID"
)

# Define RAG patterns to test
rag_patterns = [
    {"retrieval_k": 3, "rerank": False, "context_window": 2000},
    {"retrieval_k": 5, "rerank": True, "context_window": 3000},
    {"retrieval_k": 7, "rerank": True, "context_window": 4000}
]

# Run optimization experiment
results = rag_optimizer.run(
    vectorstore=vectorstore,
    patterns=rag_patterns,
    test_queries=test_queries,
    metrics=["answer_correctness", "faithfulness", "context_recall"]
)

# Display summary
print(rag_optimizer.summary())`,
    },
    {
      type: "markdown",
      content: "## Step 7: Evaluation Results\nPattern ranking based on answer_correctness, faithfulness, and context_recall metrics",
    },
    {
      type: "code",
      content: `import matplotlib.pyplot as plt

# Display leaderboard
leaderboard = [
    {"pattern": "Pattern 3", "answer_correctness": 0.7917, "faithfulness": 0.7200, "context_recall": 0.8333},
    {"pattern": "Pattern 1", "answer_correctness": 0.7292, "faithfulness": 0.6800, "context_recall": 0.7500},
    {"pattern": "Pattern 2", "answer_correctness": 0.6459, "faithfulness": 0.6000, "context_recall": 0.6900}
]

# Visualize results
fig, ax = plt.subplots(figsize=(10, 6))
patterns = [p["pattern"] for p in leaderboard]
metrics = ["answer_correctness", "faithfulness", "context_recall"]

x = range(len(patterns))
width = 0.25

for i, metric in enumerate(metrics):
    values = [p[metric] for p in leaderboard]
    ax.bar([xi + i*width for xi in x], values, width, label=metric)

ax.set_ylabel('Score')
ax.set_title('RAG Pattern Performance Comparison')
ax.set_xticks([xi + width for xi in x])
ax.set_xticklabels(patterns)
ax.legend()
plt.show()

print(f"Best pattern: {leaderboard[0]['pattern']} with avg score {sum(leaderboard[0].values()) / 3:.4f}")`,
    },
  ],
  finetuning: [
    {
      type: "markdown",
      content: "# Model Customization Pipeline\nComplete workflow for fine-tuning with synthetic data generation using InstructLab and LoRA",
    },
    {
      type: "code",
      content: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split`,
    },
    {
      type: "markdown",
      content: "## Step 1: Data Processing with Data Prep Kit (DPK)\nSimplifies document processing and parsing into AI-readable data",
    },
    {
      type: "code",
      content: `def process_documents(file_path):
    """Load and process documents using Data Prep Kit"""
    df = pd.read_csv(file_path)
    
    # Parse data into AI-readable format using DPK
    processed_data = []
    for idx, row in df.iterrows():
        processed_data.append({
            'text': row['content'],
            'metadata': {'source': row['source']},
            'category': row.get('category', 'general')
        })
    
    # Save as JSONL format
    output_path = 'processed_data.jsonl'
    with open(output_path, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\\n')
    
    print(f"✓ Processed {len(processed_data)} documents with DPK")
    print(f"✓ Saved to {output_path} in JSONL format")
    return processed_data

data = process_documents('raw_documents.csv')`,
    },
    {
      type: "markdown",
      content: "## Step 2: Synthetic Data Generation with InstructLab\nGenerate high-quality training examples using taxonomy-guided generation",
    },
    {
      type: "code",
      content: `from instructlab.sdg import generate_data

def generate_synthetic_with_ilab(base_data, taxonomy_path, num_samples=10000):
    """Generate synthetic training examples using InstructLab"""
    
    # Configure InstructLab generator
    config = {
        'taxonomy_path': taxonomy_path,
        'teacher_model': 'granite-7b-lab',
        'num_samples': num_samples,
        'temperature': 0.8,
        'diversity_penalty': 0.5,
        'multilingual': True
    }
    
    # Generate Q&A pairs using taxonomy-guided approach
    print("Generating synthetic data with InstructLab (ilab)...")
    synthetic_examples = generate_data(
        seed_data=base_data,
        taxonomy=config['taxonomy_path'],
        num_samples=config['num_samples']
    )
    
    print(f"✓ Generated {len(synthetic_examples)} synthetic Q&A pairs")
    print(f"✓ Method: Taxonomy-guided generation")
    print(f"✓ Output format: 10,000 Q&A pairs")
    
    return synthetic_examples

synthetic_data = generate_synthetic_with_ilab(
    base_data=data,
    taxonomy_path='./taxonomy',
    num_samples=10000
)`,
    },
    {
      type: "markdown",
      content: "## Step 3: Training Hub with LoRA\nFine-tune granite-7b-lab model using LoRA (Low-Rank Adaptation)",
    },
    {
      type: "code",
      content: `from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA parameters for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # Rank
    lora_alpha=16,          # Alpha scaling
    target_modules=["q_proj", "v_proj"],  # Query and Value projection layers
    lora_dropout=0.05,
    bias="none",
    inference_mode=False
)

# Load base model
print("Loading granite-7b-lab base model...")
model = AutoModelForCausalLM.from_pretrained(
    "ibm/granite-7b-lab",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = get_peft_model(model, lora_config)

# Training configuration
training_args = {
    'num_epochs': 3,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'warmup_steps': 100,
    'gradient_accumulation_steps': 4
}

print(f"✓ Model: granite-7b-lab")
print(f"✓ Method: LoRA (Low-Rank Adaptation)")
print(f"✓ Rank (r): 8, Alpha: 16")
print(f"✓ Target modules: query, value projections")
print(f"✓ Training epochs: 3")

# Training loop with progress
for epoch in range(training_args['num_epochs']):
    print(f"Epoch {epoch+1}/{training_args['num_epochs']}")
    # Training logic here
    
print("✓ Training completed successfully!")`,
    },
    {
      type: "markdown",
      content: "## Step 4: Evaluations with RAGAS + MT-Bench\nComprehensive evaluation using multiple metrics and benchmarks",
    },
    {
      type: "code",
      content: `from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def evaluate_model_comprehensive(model, test_data):
    """Run comprehensive model evaluation with RAGAS and MT-Bench"""
    predictions = []
    ground_truth = []
    
    # Generate predictions
    for item in test_data:
        pred = model.generate(item['input'])
        predictions.append(pred)
        ground_truth.append(item['label'])
    
    # Calculate RAGAS metrics
    ragas_results = evaluate(
        predictions=predictions,
        references=ground_truth,
        metrics=[answer_relevancy, faithfulness, context_recall]
    )
    
    # Calculate traditional metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'f1_score': f1_score(ground_truth, predictions, average='weighted'),
        'bleu_score': 0.7845,  # Calculated separately
        'rouge_l': 0.8123,
        'ragas_faithfulness': ragas_results['faithfulness'],
        'ragas_relevancy': ragas_results['answer_relevancy']
    }
    
    print(f"✓ Framework: RAGAS + MT-Bench")
    print(f"✓ Metrics: BLEU ({metrics['bleu_score']:.4f}), ROUGE-L ({metrics['rouge_l']:.4f}), F1 ({metrics['f1_score']:.4f})")
    print(f"✓ Benchmark dataset: sentiment-test-1k")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics

results = evaluate_model_comprehensive(model, test_data)
print(f"\\n📊 Evaluation Results: {results}")`,
    },
  ],
  synthetic: [
    {
      type: "markdown",
      content: "# Synthetic Data Generation Pipeline\nFocused workflow for generating high-quality synthetic training data using InstructLab",
    },
    {
      type: "code",
      content: `import pandas as pd
import json
from instructlab.sdg import generate_data, SDGConfig
from sklearn.model_selection import train_test_split`,
    },
    {
      type: "markdown",
      content: "## Step 1: Load Seed Dataset\nLoad 50 seed examples to guide synthetic data generation",
    },
    {
      type: "code",
      content: `def load_seed_examples(file_path, num_samples=50):
    """Load seed examples for synthetic generation"""
    df = pd.read_csv(file_path)
    
    # Sample seed examples
    seed_data = df.sample(n=num_samples, random_state=42)
    
    # Convert to required format
    seed_examples = []
    for idx, row in seed_data.iterrows():
        seed_examples.append({
            'input': row['text'],
            'category': row['category'],
            'label': row['label']
        })
    
    print(f"✓ Loaded {len(seed_examples)} seed examples")
    print(f"✓ Categories: {seed_data['category'].nunique()} classes")
    print(f"✓ Distribution: {seed_data['category'].value_counts().to_dict()}")
    
    return seed_examples

seed_data = load_seed_examples('training_data.csv', num_samples=50)`,
    },
    {
      type: "markdown",
      content: "## Step 2: Define Taxonomy Structure\nPrepare taxonomy for guided synthetic data generation",
    },
    {
      type: "code",
      content: `# Define taxonomy structure for InstructLab
taxonomy = {
    'version': 1,
    'domain': 'classification',
    'categories': [
        'technology',
        'healthcare',
        'finance',
        'education',
        'entertainment'
    ],
    'seed_examples': seed_data,
    'generation_config': {
        'diversity_target': 0.85,
        'quality_threshold': 0.7,
        'multilingual': ['en', 'es', 'fr']
    }
}

# Save taxonomy
with open('taxonomy.json', 'w') as f:
    json.dump(taxonomy, f, indent=2)

print("✓ Taxonomy defined with 5 categories")
print("✓ Multilingual support: English, Spanish, French")`,
    },
    {
      type: "markdown",
      content: "## Step 3: Configure InstructLab Generator\nSet up synthetic data generation parameters",
    },
    {
      type: "code",
      content: `# Configure InstructLab synthetic data generator
sdg_config = SDGConfig(
    model="granite-7b-lab",
    temperature=0.7,
    diversity_penalty=0.3,
    num_samples=5000,
    quality_threshold=0.7,
    multilingual=True,
    languages=['en', 'es', 'fr']
)

print("📋 Generation Configuration:")
print(f"  - Temperature: {sdg_config.temperature}")
print(f"  - Diversity penalty: {sdg_config.diversity_penalty}")
print(f"  - Target samples: {sdg_config.num_samples}")
print(f"  - Quality threshold: {sdg_config.quality_threshold}")
print(f"  - Languages: {', '.join(sdg_config.languages)}")`,
    },
    {
      type: "markdown",
      content: "## Step 4: Run Synthetic Data Generation\nGenerate 5,000 high-quality synthetic examples with progress tracking",
    },
    {
      type: "code",
      content: `from tqdm import tqdm

# Initialize generator
generator = generate_data(
    config=sdg_config,
    taxonomy='taxonomy.json',
    seed_data=seed_data
)

# Generate synthetic data with progress tracking
synthetic_examples = []
print("\\n🚀 Starting synthetic data generation...")

with tqdm(total=sdg_config.num_samples, desc="Generating") as pbar:
    for example in generator:
        synthetic_examples.append(example)
        pbar.update(1)
        
        # Update metrics every 100 samples
        if len(synthetic_examples) % 100 == 0:
            diversity_score = calculate_diversity(synthetic_examples)
            quality_score = calculate_quality(synthetic_examples)
            pbar.set_postfix({
                'diversity': f'{diversity_score:.2f}',
                'quality': f'{quality_score:.2f}'
            })

print(f"\\n✓ Generated {len(synthetic_examples)} synthetic samples")
print(f"✓ Final diversity score: 0.85")
print(f"✓ Quality threshold maintained: >0.7")`,
    },
    {
      type: "markdown",
      content: "## Step 5: Quality Filtering and Validation\nFilter and validate generated synthetic data",
    },
    {
      type: "code",
      content: `def filter_synthetic_data(examples, quality_threshold=0.7):
    """Filter synthetic examples based on quality score"""
    filtered = []
    
    for example in examples:
        if example['quality_score'] >= quality_threshold:
            filtered.append(example)
    
    print(f"✓ Filtered: {len(filtered)}/{len(examples)} examples passed quality check")
    print(f"✓ Pass rate: {len(filtered)/len(examples)*100:.1f}%")
    
    return filtered

# Apply quality filter
high_quality_data = filter_synthetic_data(synthetic_examples, quality_threshold=0.7)

# Validate distribution
categories = [e['category'] for e in high_quality_data]
print(f"\\n📊 Category Distribution:")
for cat in set(categories):
    count = categories.count(cat)
    print(f"  - {cat}: {count} samples ({count/len(categories)*100:.1f}%)")`,
    },
    {
      type: "markdown",
      content: "## Step 6: Export Synthetic Dataset\nSave generated data to file for downstream tasks",
    },
    {
      type: "code",
      content: `# Export to JSONL format
output_file = 'synthetic_training_data.jsonl'
with open(output_file, 'w') as f:
    for example in high_quality_data:
        f.write(json.dumps(example) + '\\n')

print(f"\\n✅ Synthetic dataset exported successfully!")
print(f"✓ File: {output_file}")
print(f"✓ Total samples: {len(high_quality_data)}")
print(f"✓ Format: JSONL")

# Export to CSV as well
df = pd.DataFrame(high_quality_data)
df.to_csv('synthetic_training_data.csv', index=False)
print(f"✓ Also saved as CSV for convenience")`,
    },
    {
      type: "markdown",
      content: "## Step 7: Data Distribution Analysis\nVisualize the generated synthetic dataset",
    },
    {
      type: "code",
      content: `import matplotlib.pyplot as plt
import seaborn as sns

# Category distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Category distribution
category_counts = df['category'].value_counts()
axes[0, 0].bar(category_counts.index, category_counts.values, color='steelblue')
axes[0, 0].set_title('Samples per Category')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Quality score distribution
axes[0, 1].hist(df['quality_score'], bins=20, color='green', alpha=0.7)
axes[0, 1].axvline(0.7, color='red', linestyle='--', label='Threshold')
axes[0, 1].set_title('Quality Score Distribution')
axes[0, 1].set_xlabel('Quality Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Plot 3: Language distribution
language_counts = df['language'].value_counts()
axes[1, 0].pie(language_counts.values, labels=language_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title('Language Distribution')

# Plot 4: Text length distribution
text_lengths = df['text'].apply(len)
axes[1, 1].hist(text_lengths, bins=30, color='orange', alpha=0.7)
axes[1, 1].set_title('Text Length Distribution')
axes[1, 1].set_xlabel('Character Count')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("\\n📈 Analysis complete! Synthetic dataset is ready for training.")`,
    },
  ],
  llmserving: [
    {
      type: "markdown",
      content: "# Distributed LLM Serving with llm-d\nDeploy Llama-3-70B on Red Hat OpenShift AI using llm-d for high-throughput, low-latency inference",
    },
    {
      type: "code",
      content: `import os
from kubernetes import client, config
from kserve import KServeClient, V1beta1InferenceService
import yaml`,
    },
    {
      type: "markdown",
      content: "## Step 1: Model Configuration\nConfigure Llama-3-70B with tensor parallelism across multiple GPUs",
    },
    {
      type: "code",
      content: `# Model configuration for distributed serving
model_config = {
    "model_id": "meta-llama/Llama-3-70B-Instruct",
    "tensor_parallel_size": 4,  # Distribute across 4 GPUs per node
    "dtype": "float16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.90,
    "quantization": None,  # FP16 for quality, use "awq" for memory savings
}

print("✓ Model: Llama-3-70B-Instruct")
print(f"✓ Tensor Parallel Size: {model_config['tensor_parallel_size']} GPUs")
print(f"✓ Precision: {model_config['dtype'].upper()}")
print(f"✓ Max Context Length: {model_config['max_model_len']} tokens")
print(f"✓ GPU Memory Utilization: {model_config['gpu_memory_utilization']*100}%")`,
    },
    {
      type: "markdown",
      content: "## Step 2: llm-d Distributed Setup\nConfigure llm-d with disaggregated prefill and decode pools",
    },
    {
      type: "code",
      content: `# llm-d configuration for disaggregated serving
llmd_config = """
apiVersion: llm-d.ai.redhat.com/v1alpha1
kind: LLMDeployment
metadata:
  name: llama3-70b-distributed
  namespace: llm-serving
spec:
  model:
    id: meta-llama/Llama-3-70B-Instruct
    source: huggingface
  
  # Disaggregated architecture for optimal throughput
  prefillPool:
    replicas: 2
    resources:
      limits:
        nvidia.com/gpu: 4
        memory: "256Gi"
    tensorParallelSize: 4
    
  decodePool:
    replicas: 4
    resources:
      limits:
        nvidia.com/gpu: 4
        memory: "256Gi"
    tensorParallelSize: 4
    
  # vLLM backend configuration
  backend:
    type: vllm
    maxModelLen: 8192
    gpuMemoryUtilization: 0.9
"""

print("✓ llm-d Deployment Configuration:")
print("  - Prefill Workers: 2 replicas × 4 GPUs = 8 GPUs")
print("  - Decode Workers: 4 replicas × 4 GPUs = 16 GPUs")
print("  - Total GPU Allocation: 24 GPUs")
print("  - Backend: vLLM with continuous batching")`,
    },
    {
      type: "markdown",
      content: "## Step 3: KServe Integration\nWrap llm-d deployment in KServe InferenceService for OpenShift AI integration",
    },
    {
      type: "code",
      content: `# KServe InferenceService configuration
kserve_config = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama3-70b-service
  namespace: llm-serving
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
    sidecar.istio.io/inject: "true"
spec:
  predictor:
    minReplicas: 2
    maxReplicas: 8
    scaleTarget: 70  # Target 70% GPU utilization
    scaleMetric: gpu
    
    containers:
    - name: kserve-container
      image: quay.io/rhoai/llm-d-server:latest
      args:
        - --model-id=meta-llama/Llama-3-70B-Instruct
        - --tensor-parallel-size=4
        - --enable-prefix-caching
        - --max-model-len=8192
      resources:
        limits:
          nvidia.com/gpu: 4
          memory: "256Gi"
        requests:
          nvidia.com/gpu: 4
          memory: "200Gi"
      
      ports:
      - containerPort: 8080
        protocol: TCP
        name: http
      - containerPort: 8081
        protocol: TCP
        name: grpc
"""

# Apply KServe configuration
print("✓ KServe InferenceService configured:")
print("  - Min Replicas: 2")
print("  - Max Replicas: 8 (autoscaling enabled)")
print("  - Scale Metric: GPU utilization @ 70%")
print("  - Endpoints: REST (8080) + gRPC (8081)")`,
    },
    {
      type: "markdown",
      content: "## Step 4: Prefix Cache Configuration\nEnable prefix caching to optimize latency for repeated prompt patterns",
    },
    {
      type: "code",
      content: `# Prefix caching configuration for llm-d
prefix_cache_config = {
    "enable_prefix_caching": True,
    "prefix_cache_size_gb": 32,
    "block_size": 16,
    "eviction_policy": "lru",
    "hash_algorithm": "sha256",
    "cache_common_prefixes": [
        "You are a helpful enterprise assistant...",
        "Based on our company documentation...",
        "Analyze the following customer query..."
    ]
}

# Expected improvements
print("✓ Prefix Caching Enabled:")
print(f"  - Cache Size: {prefix_cache_config['prefix_cache_size_gb']}GB")
print(f"  - Block Size: {prefix_cache_config['block_size']} tokens")
print(f"  - Eviction Policy: {prefix_cache_config['eviction_policy'].upper()}")
print("\\n📊 Expected Performance Improvements:")
print("  - Time-to-first-token: -60% for cached prefixes")
print("  - Throughput: +40% for similar prompts")
print("  - Memory efficiency: Shared KV-cache across requests")`,
    },
    {
      type: "markdown",
      content: "## Step 5: Load Balancing Configuration\nConfigure Istio service mesh for intelligent request routing",
    },
    {
      type: "code",
      content: `# Istio Gateway and VirtualService configuration
istio_config = """
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: llm-gateway
  namespace: llm-serving
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: llm-tls-secret
    hosts:
    - llm.apps.openshift.example.com
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llama3-routing
  namespace: llm-serving
spec:
  hosts:
  - llm.apps.openshift.example.com
  gateways:
  - llm-gateway
  http:
  - match:
    - uri:
        prefix: /v1/completions
    route:
    - destination:
        host: llama3-70b-service
        port:
          number: 8080
    retries:
      attempts: 3
      perTryTimeout: 30s
    timeout: 120s
"""

print("✓ Istio Load Balancing Configured:")
print("  - Gateway: HTTPS with TLS termination")
print("  - Session Affinity: Enabled for streaming")
print("  - Retry Policy: 3 attempts, 30s per try")
print("  - Request Timeout: 120s max")`,
    },
    {
      type: "markdown",
      content: "## Step 6: Monitoring & Observability\nIntegrate Prometheus metrics and Grafana dashboards",
    },
    {
      type: "code",
      content: `# Prometheus ServiceMonitor for llm-d metrics
monitoring_config = """
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llama3-metrics
  namespace: llm-serving
spec:
  selector:
    matchLabels:
      app: llama3-70b-service
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
"""

# Key metrics to track
llm_metrics = {
    "vllm_num_requests_running": "Active requests being processed",
    "vllm_num_requests_waiting": "Queued requests",
    "vllm_gpu_cache_usage_perc": "GPU KV-cache utilization",
    "vllm_avg_prompt_throughput_toks_per_s": "Input token throughput",
    "vllm_avg_generation_throughput_toks_per_s": "Output token throughput",
    "vllm_request_latency_seconds": "End-to-end request latency",
    "vllm_time_to_first_token_seconds": "Time to first token (TTFT)",
}

print("✓ Prometheus Metrics Configured:")
print("  - Scrape Interval: 15s")
for metric, desc in list(llm_metrics.items())[:5]:
    print(f"  - {metric}: {desc}")

print("\\n📊 Grafana Dashboard Panels:")
print("  - Token Throughput (input/output)")
print("  - Request Latency P50/P95/P99")
print("  - GPU Utilization & Cache Usage")
print("  - Queue Depth & Active Requests")`,
    },
    {
      type: "markdown",
      content: "## Step 7: Deploy and Test\nApply configuration and validate the deployment",
    },
    {
      type: "code",
      content: `# Deploy llm-d service
def deploy_llm_service():
    """Deploy the complete llm-d stack to OpenShift AI"""
    
    # Apply configurations
    steps = [
        ("Creating namespace...", "oc create namespace llm-serving"),
        ("Applying llm-d deployment...", "oc apply -f llmd-deployment.yaml"),
        ("Configuring KServe...", "oc apply -f kserve-inference.yaml"),
        ("Setting up Istio routing...", "oc apply -f istio-config.yaml"),
        ("Enabling monitoring...", "oc apply -f servicemonitor.yaml"),
    ]
    
    for step, cmd in steps:
        print(f"\\n{step}")
        print(f"  $ {cmd}")
        print("  ✓ Applied successfully")
    
    return True

# Run deployment
deploy_llm_service()

# Test inference endpoint
print("\\n🧪 Testing Inference Endpoint...")
test_payload = {
    "model": "llama3-70b",
    "messages": [{"role": "user", "content": "Hello, how can you help me today?"}],
    "max_tokens": 100
}
print(f"  POST https://llm.apps.openshift.example.com/v1/chat/completions")
print(f"  Payload: {test_payload}")
print("\\n✅ Deployment Complete!")
print("  - Endpoint: https://llm.apps.openshift.example.com")
print("  - Status: Ready (2/2 replicas)")
print("  - Avg Latency: 245ms TTFT, 42 tokens/s generation")`,
    },
  ],
  agentic: [
    {
      type: "markdown",
      content: "# GitHub Repository Maintenance Agent Starter Kit\nAn agentic workflow for automated issue triage, PR review suggestions, labeling, and dependency management",
    },
    {
      type: "code",
      content: `import os
from github import Github
from openai import OpenAI
import json
from datetime import datetime, timedelta`,
    },
    {
      type: "markdown",
      content: "## Step 1: GitHub API Connection\nEstablish secure connection to GitHub API using fine-grained PAT",
    },
    {
      type: "code",
      content: `# Initialize GitHub client with fine-grained PAT
def connect_github():
    """Establish authenticated connection to GitHub API"""
    
    # Load token from OpenShift Secrets
    github_token = os.environ.get("GITHUB_PAT")
    
    if not github_token:
        raise ValueError("GITHUB_PAT not found in environment")
    
    gh = Github(github_token)
    
    # Verify authentication
    user = gh.get_user()
    print(f"✓ Connected to GitHub as: {user.login}")
    print(f"✓ Rate limit remaining: {gh.rate_limiting[0]}/{gh.rate_limiting[1]}")
    
    return gh

# Connect to GitHub
gh_client = connect_github()

# Get target repository
repo = gh_client.get_repo("example-org/sample-repository")
print(f"✓ Repository: {repo.full_name}")
print(f"✓ Open Issues: {repo.open_issues_count}")
print(f"✓ Open PRs: {len(list(repo.get_pulls(state='open')))}")`,
    },
    {
      type: "markdown",
      content: "## Step 2: Define Agent Tools\nCreate modular tools for issue triage, labeling, PR review suggestion, and dependency checks",
    },
    {
      type: "code",
      content: `# Initialize LLM client for agent reasoning
llm_client = OpenAI(base_url="https://llm.apps.openshift.example.com/v1")

# Tool: Issue Triage and Classification
def triage_issue(issue):
    """Analyze issue and determine category and priority"""
    
    prompt = f"""Analyze this GitHub issue and classify it:
    
Title: {issue.title}
Body: {issue.body[:1000] if issue.body else 'No description'}

Respond with JSON:
{{"category": "bug|enhancement|question|documentation", "priority": "low|medium|high", "reasoning": "..."}}"""

    response = llm_client.chat.completions.create(
        model="llama3-70b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Tool: Auto-labeling
def apply_labels(issue, classification):
    """Apply appropriate labels based on classification"""
    
    label_map = {
        "bug": ["bug", "needs-triage"],
        "enhancement": ["enhancement", "feature-request"],
        "question": ["question", "help-wanted"],
        "documentation": ["documentation", "good-first-issue"]
    }
    
    priority_labels = {
        "high": "priority/critical",
        "medium": "priority/normal", 
        "low": "priority/low"
    }
    
    labels = label_map.get(classification["category"], [])
    labels.append(priority_labels.get(classification["priority"], "priority/normal"))
    
    issue.add_to_labels(*labels)
    return labels

print("✓ Defined tools: triage_issue, apply_labels")`,
    },
    {
      type: "markdown",
      content: "## Step 3: PR Review Suggestion Tool\nSuggest reviewers based on CODEOWNERS and file change analysis",
    },
    {
      type: "code",
      content: `# Tool: PR Review Suggestion
def suggest_reviewers(pull_request):
    """Suggest reviewers based on CODEOWNERS and modified files"""
    
    # Parse CODEOWNERS file
    try:
        codeowners_content = repo.get_contents("CODEOWNERS").decoded_content.decode()
        codeowners = parse_codeowners(codeowners_content)
    except:
        codeowners = {}
    
    # Get modified files
    modified_files = [f.filename for f in pull_request.get_files()]
    
    # Find matching owners
    suggested_reviewers = set()
    for file_path in modified_files:
        for pattern, owners in codeowners.items():
            if match_pattern(file_path, pattern):
                suggested_reviewers.update(owners)
    
    # Exclude PR author
    suggested_reviewers.discard(pull_request.user.login)
    
    # Request reviews
    if suggested_reviewers:
        pull_request.create_review_request(reviewers=list(suggested_reviewers)[:3])
    
    return list(suggested_reviewers)

def parse_codeowners(content):
    """Parse CODEOWNERS file into pattern-owners mapping"""
    owners = {}
    for line in content.split("\\n"):
        if line.strip() and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                pattern = parts[0]
                owners[pattern] = [o.lstrip("@") for o in parts[1:]]
    return owners

print("✓ Defined tool: suggest_reviewers with CODEOWNERS support")`,
    },
    {
      type: "markdown",
      content: "## Step 4: Dependency Check Tool\nDetect CI failures due to outdated packages and create upgrade tasks",
    },
    {
      type: "code",
      content: `# Tool: Dependency Check
def check_dependencies(check_run):
    """Analyze CI failure and create dependency upgrade issues if needed"""
    
    if check_run.conclusion != "failure":
        return None
    
    # Get check run logs (simplified)
    logs = get_check_run_logs(check_run)
    
    # Analyze for dependency issues
    prompt = f"""Analyze this CI failure log and determine if it's caused by outdated dependencies:

{logs[:2000]}

Respond with JSON:
{{"is_dependency_issue": true/false, "packages": ["pkg1", "pkg2"], "recommended_versions": {{"pkg1": "1.2.3"}}, "reasoning": "..."}}"""

    response = llm_client.chat.completions.create(
        model="llama3-70b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    analysis = json.loads(response.choices[0].message.content)
    
    if analysis["is_dependency_issue"]:
        # Create follow-up issue
        issue_body = f"""## Dependency Upgrade Required

CI build failed due to outdated packages.

**Affected packages:**
{chr(10).join(f"- {pkg}: upgrade to {analysis['recommended_versions'].get(pkg, 'latest')}" for pkg in analysis['packages'])}

**Analysis:** {analysis['reasoning']}

**Related check run:** {check_run.html_url}

---
_This issue was automatically created by the GitHub Maintenance Agent._
"""
        
        new_issue = repo.create_issue(
            title=f"[Auto] Dependency upgrade: {', '.join(analysis['packages'][:3])}",
            body=issue_body,
            labels=["dependencies", "automated", "priority/high"]
        )
        return new_issue
    
    return None

print("✓ Defined tool: check_dependencies with CI log analysis")`,
    },
    {
      type: "markdown",
      content: "## Step 5: End-to-End Agent Simulation\nSimulate agent run on sample repository events",
    },
    {
      type: "code",
      content: `# Simulate agent processing
def run_agent_simulation(repo):
    """Simulate end-to-end agent run on sample events"""
    
    results = {
        "issues_processed": 0,
        "prs_processed": 0,
        "labels_applied": [],
        "reviewers_suggested": [],
        "dependency_issues_created": 0,
        "decisions": []
    }
    
    # Process open issues
    print("\\n🔍 Processing open issues...")
    for issue in list(repo.get_issues(state="open"))[:5]:
        if issue.pull_request:
            continue
            
        classification = triage_issue(issue)
        labels = apply_labels(issue, classification)
        
        results["issues_processed"] += 1
        results["labels_applied"].extend(labels)
        results["decisions"].append({
            "type": "issue_triage",
            "issue": issue.number,
            "classification": classification,
            "action": f"Applied labels: {labels}"
        })
        
        print(f"  Issue #{issue.number}: {classification['category']} ({classification['priority']})")
    
    # Process open PRs
    print("\\n🔍 Processing open pull requests...")
    for pr in list(repo.get_pulls(state="open"))[:5]:
        reviewers = suggest_reviewers(pr)
        
        results["prs_processed"] += 1
        results["reviewers_suggested"].extend(reviewers)
        results["decisions"].append({
            "type": "pr_review",
            "pr": pr.number,
            "action": f"Suggested reviewers: {reviewers}"
        })
        
        print(f"  PR #{pr.number}: Suggested reviewers: {reviewers}")
    
    print(f"\\n✅ Agent Simulation Complete!")
    print(f"  - Issues processed: {results['issues_processed']}")
    print(f"  - PRs processed: {results['prs_processed']}")
    print(f"  - Total labels applied: {len(results['labels_applied'])}")
    print(f"  - Reviewers suggested: {len(set(results['reviewers_suggested']))}")
    
    return results

# Run simulation
simulation_results = run_agent_simulation(repo)`,
    },
    {
      type: "markdown",
      content: "## Step 6: Decision Logging\nLog all agent decisions for monitoring and iteration",
    },
    {
      type: "code",
      content: `import json
from datetime import datetime

# Decision logging for observability
class DecisionLogger:
    def __init__(self, log_path="agent_decisions.jsonl"):
        self.log_path = log_path
        self.decisions = []
    
    def log(self, decision_type, context, action, reasoning):
        """Log an agent decision with full context"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": decision_type,
            "context": context,
            "action": action,
            "reasoning": reasoning
        }
        
        self.decisions.append(entry)
        
        # Append to JSONL file
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\\n")
        
        return entry
    
    def get_metrics(self):
        """Calculate metrics from logged decisions"""
        return {
            "total_decisions": len(self.decisions),
            "by_type": self._count_by_type(),
            "avg_response_time_ms": 245  # Placeholder
        }
    
    def _count_by_type(self):
        counts = {}
        for d in self.decisions:
            counts[d["type"]] = counts.get(d["type"], 0) + 1
        return counts

# Initialize logger
logger = DecisionLogger()

# Log simulation decisions
for decision in simulation_results["decisions"]:
    logger.log(
        decision_type=decision["type"],
        context={"issue": decision.get("issue"), "pr": decision.get("pr")},
        action=decision["action"],
        reasoning=decision.get("classification", {}).get("reasoning", "Based on CODEOWNERS")
    )

print("\\n📊 Decision Metrics:")
metrics = logger.get_metrics()
print(f"  - Total decisions logged: {metrics['total_decisions']}")
for dtype, count in metrics['by_type'].items():
    print(f"  - {dtype}: {count}")`,
    },
    {
      type: "markdown",
      content: "## Step 7: Deploy Agent Configuration\nConfigure webhook routing and scheduled execution",
    },
    {
      type: "code",
      content: `# Agent deployment configuration
agent_config = """
apiVersion: agents.ai.redhat.com/v1alpha1
kind: GitHubMaintenanceAgent
metadata:
  name: repo-maintenance-agent
  namespace: github-agents
spec:
  repository:
    owner: example-org
    name: sample-repository
  
  # GitHub webhook configuration
  webhook:
    events:
      - issues.opened
      - issues.reopened
      - pull_request.opened
      - pull_request.synchronize
      - check_run.completed
    secret:
      name: github-webhook-secret
      key: webhook-secret
  
  # Scheduled execution for batch operations
  schedule:
    # Run every 15 minutes for new events
    interval: "*/15 * * * *"
    # Daily cleanup of stale issues
    staleIssueCleanup: "0 2 * * *"
  
  # Secrets configuration
  secrets:
    githubToken:
      name: github-pat-secret
      key: token
    llmEndpoint:
      name: llm-endpoint-secret
      key: api-key
  
  # Monitoring
  metrics:
    enabled: true
    endpoint: /metrics
    port: 8080
"""

print("✓ Agent Configuration Generated:")
print("  - Webhook events: issues, pull_requests, check_runs")
print("  - Schedule: Every 15 minutes + daily cleanup at 2 AM")
print("  - Secrets: Managed via OpenShift Secrets")
print("  - Metrics: Prometheus-compatible on :8080/metrics")

print("\\n🚀 Ready to deploy with: oc apply -f agent-config.yaml")`,
    },
  ],
};

interface NotebookViewProps {
  pipelineType: PipelineType;
}

export const NotebookView = ({ pipelineType }: NotebookViewProps) => {
  const notebookCells = notebookConfigs[pipelineType];
  const explanation = notebookExplanations[pipelineType];

  return (
    <div className="h-full flex flex-col bg-background">
      <div className="p-4 border-b border-border flex items-center justify-between bg-card">
        <h3 className="font-semibold">
          {pipelineType === "rag" ? "rag_optimization_pipeline.ipynb" : 
           pipelineType === "synthetic" ? "synthetic_data_generation.ipynb" : 
           pipelineType === "llmserving" ? "llm_distributed_serving.ipynb" :
           pipelineType === "agentic" ? "github_maintenance_agent.ipynb" :
           "model_customization_pipeline.ipynb"}
        </h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Play className="w-4 h-4" />
            Run All
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="w-4 h-4" />
            Download .ipynb
          </Button>
          {pipelineType === "synthetic" && (
            <Button size="sm" className="gap-2 bg-green-600 hover:bg-green-700">
              Continue to Training
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Explainability Section */}
        <div className="border border-[hsl(var(--notebook-border))] rounded-md overflow-hidden bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30">
          <div className="flex items-center gap-2 px-3 py-2 bg-amber-100/50 dark:bg-amber-900/30 text-sm font-medium border-b border-amber-200 dark:border-amber-800">
            <Lightbulb className="w-4 h-4 text-amber-600 dark:text-amber-400" />
            <span className="text-amber-800 dark:text-amber-300">Explain this recommendation</span>
          </div>
          <div className="p-4 space-y-3">
            <ul className="space-y-2">
              {explanation.bullets.map((bullet, idx) => (
                <li key={idx} className="flex items-start gap-2 text-sm text-foreground">
                  <span className="text-amber-600 dark:text-amber-400 mt-1">•</span>
                  <span dangerouslySetInnerHTML={{ 
                    __html: bullet.replace(/\*\*(.*?)\*\*/g, '<strong class="text-amber-700 dark:text-amber-300">$1</strong>') 
                  }} />
                </li>
              ))}
            </ul>
            <div className="flex items-center gap-2 pt-2 border-t border-amber-200 dark:border-amber-800">
              <BookOpen className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground italic">{explanation.citation}</span>
            </div>
          </div>
        </div>

        {notebookCells.map((cell, index) => (
          <div key={index} className="border border-[hsl(var(--notebook-border))] rounded-md overflow-hidden">
            <div className="flex items-center px-3 py-1 bg-muted text-xs text-muted-foreground border-b border-[hsl(var(--notebook-border))]">
              <span className="font-mono">[{index + 1}]</span>
            </div>
            <div className="p-4">
              {cell.type === "markdown" ? (
                <div className="prose prose-sm max-w-none">
                  {cell.content.split("\n").map((line, i) => {
                    if (line.startsWith("# ")) {
                      return (
                        <h1 key={i} className="text-2xl font-bold mb-2">
                          {line.substring(2)}
                        </h1>
                      );
                    }
                    if (line.startsWith("## ")) {
                      return (
                        <h2 key={i} className="text-xl font-semibold mb-2 mt-4">
                          {line.substring(3)}
                        </h2>
                      );
                    }
                    return (
                      <p key={i} className="text-sm text-muted-foreground">
                        {line}
                      </p>
                    );
                  })}
                </div>
              ) : (
                <pre className="bg-[hsl(var(--code-bg))] p-4 rounded text-sm font-mono overflow-x-auto">
                  <code>{cell.content}</code>
                </pre>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
