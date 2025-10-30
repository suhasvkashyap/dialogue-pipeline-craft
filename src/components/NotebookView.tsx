import { Download, Play } from "lucide-react";
import { Button } from "@/components/ui/button";

const notebookCells = [
  {
    type: "markdown",
    content: "# Model Customization Pipeline\nComplete workflow for fine-tuning with synthetic data generation",
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
    content: "## Step 1: Data Processing\nSimplifies document processing and parsing into AI-readable data",
  },
  {
    type: "code",
    content: `def process_documents(file_path):
    """Load and process documents into structured format"""
    df = pd.read_csv(file_path)
    
    # Parse data into AI-readable format
    processed_data = []
    for idx, row in df.iterrows():
        processed_data.append({
            'text': row['content'],
            'metadata': {'source': row['source']}
        })
    
    # Save processed data
    output_path = 'processed_data.json'
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
    
    return processed_data

data = process_documents('raw_documents.csv')
print(f"Processed {len(data)} documents")`,
  },
  {
    type: "markdown",
    content: "## Step 2: Synthetic Data Generation Hub\nGenerate high-quality training examples with dynamic parameters",
  },
  {
    type: "code",
    content: `def generate_synthetic_data(base_data, num_samples=1000):
    """Generate synthetic training examples"""
    synthetic_examples = []
    
    # Generation parameters
    config = {
        'temperature': 0.8,
        'diversity_penalty': 0.5,
        'multilingual': True,
        'num_samples': num_samples
    }
    
    for i in range(num_samples):
        # Generate high-quality training examples
        example = {
            'prompt': f"Example prompt {i}",
            'completion': f"Generated completion {i}",
            'quality_score': 0.95
        }
        synthetic_examples.append(example)
    
    print(f"Generated {len(synthetic_examples)} synthetic examples")
    return synthetic_examples

synthetic_data = generate_synthetic_data(data)`,
  },
  {
    type: "markdown",
    content: "## Step 3: Training Hub\nAlgorithm-focused interface for LLM training with LoRA/QLoRA",
  },
  {
    type: "code",
    content: `from peft import LoraConfig, get_peft_model

# Configure LoRA parameters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
model = get_peft_model(model, lora_config)

# Training configuration
training_args = {
    'num_epochs': 3,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'warmup_steps': 100
}

# Training loop with progress
for epoch in range(training_args['num_epochs']):
    print(f"Epoch {epoch+1}/{training_args['num_epochs']}")
    # Training logic here
    
print("Training completed successfully!")`,
  },
  {
    type: "markdown",
    content: "## Step 4: Evaluations\nDistributed execution of evaluation jobs from popular frameworks",
  },
  {
    type: "code",
    content: `from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def evaluate_model(model, test_data):
    """Run comprehensive model evaluation"""
    predictions = []
    ground_truth = []
    
    for item in test_data:
        pred = model.generate(item['input'])
        predictions.append(pred)
        ground_truth.append(item['label'])
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(ground_truth, predictions),
        'f1_score': f1_score(ground_truth, predictions, average='weighted'),
        'perplexity': 2.45
    }
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.show()
    
    return metrics

results = evaluate_model(model, test_data)
print(f"Evaluation Results: {results}")`,
  },
];

export const NotebookView = () => {
  return (
    <div className="h-full flex flex-col bg-background">
      <div className="p-4 border-b border-border flex items-center justify-between bg-card">
        <h3 className="font-semibold">model_customization_pipeline.ipynb</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Play className="w-4 h-4" />
            Run All
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="w-4 h-4" />
            Download .ipynb
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
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
