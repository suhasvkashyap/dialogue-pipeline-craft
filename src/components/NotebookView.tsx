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

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
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
