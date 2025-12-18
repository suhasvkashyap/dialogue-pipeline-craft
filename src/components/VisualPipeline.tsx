import { Binary, Network, Laptop, BarChart3, CheckCircle2, Edit, Play, Calendar, FileText, Scissors, Grid3x3, Database, Search, Trophy, Medal, HelpCircle, BookOpen, Server, Cpu, Activity, Shield, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PipelineType } from "./ChatInterface";
import { Slider } from "@/components/ui/slider";

interface PipelineStage {
  icon: any;
  title: string;
  description: string;
  completed: boolean;
  running?: boolean;
  badges?: string[];
}

interface StageExplanation {
  stage: string;
  reason: string;
}

interface PipelineExplanation {
  summary: string;
  stages: StageExplanation[];
  citation: string;
}

const pipelineExplanations: Record<PipelineType, PipelineExplanation> = {
  rag: {
    summary: "This RAG pipeline is designed for customer support knowledge base retrieval with Mistral-7B, optimized for sub-2-second response times and source-linked answers.",
    stages: [
      { stage: "Document Upload", reason: "Ingests product documentation and support playbooks from your Elasticsearch cluster with tag filtering." },
      { stage: "Document Chunking", reason: "512-token chunks with overlap preserve context in long troubleshooting guides while enabling precise retrieval." },
      { stage: "Embedding Generation", reason: "Selected embedding model optimized for Mistral-7B compatibility and semantic search accuracy." },
      { stage: "Vector Store", reason: "Elasticsearch vector storage enables seamless integration with your existing infrastructure." },
      { stage: "Retrieval Testing", reason: "Validates retrieval accuracy before deployment to ensure agent confidence in answers." },
      { stage: "RAG Pattern Ranking", reason: "Evaluates multiple retrieval configurations to optimize answer quality and faithfulness metrics." }
    ],
    citation: "Derived from Red Hat AI validated demos – RAG on product docs (fictitious)."
  },
  finetuning: {
    summary: "This fine-tuning pipeline adapts an LLM for insurance claim email classification using LoRA, with built-in PII protection and performance validation.",
    stages: [
      { stage: "Data processing", reason: "Ingests historical emails from S3 with PII redaction and class balancing to prevent bias in model training." },
      { stage: "Synthetic data Generation hub", reason: "Augments training data with InstructLab to improve coverage across all claim categories." },
      { stage: "Training hub", reason: "LoRA fine-tuning minimizes GPU cost while achieving effective adaptation for multi-class classification." },
      { stage: "Evaluations", reason: "Macro-F1 threshold of 0.85 ensures reliable classification before model registration and deployment." }
    ],
    citation: "Recommended based on Red Hat AI Notebook examples, section \"Email Classification Fine-Tuning\" (fictitious)."
  },
  synthetic: {
    summary: "This pipeline generates privacy-preserving synthetic telco customer data for churn prediction modeling without exposing production PII.",
    stages: [
      { stage: "Data preprocessing", reason: "Analyzes source Parquet data structure and enforces k-anonymity on sensitive CDR fields." },
      { stage: "Synthetic data Generation hub", reason: "Generates 1M+ balanced records preserving usage-churn correlations with automated AUC validation." }
    ],
    citation: "Derived from Red Hat OpenShift AI documentation, section \"Synthetic Data for ML Training\" (fictitious)."
  },
  llmserving: {
    summary: "This pipeline deploys Llama-3-70B using llm-d for distributed inference on OpenShift AI, with multi-node GPU scaling, prefix caching, and KServe-compatible endpoints.",
    stages: [
      { stage: "Model Configuration", reason: "Configures Llama-3-70B with tensor parallelism across 4 GPUs per node for optimal memory distribution." },
      { stage: "llm-d Distributed Setup", reason: "Deploys llm-d disaggregated serving with separate prefill and decode pools for maximum throughput." },
      { stage: "KServe Integration", reason: "Wraps llm-d deployment in KServe InferenceService for OpenShift AI-native autoscaling and routing." },
      { stage: "Prefix Cache Config", reason: "Enables prefix caching to reuse KV-cache across requests, reducing latency for common prompt patterns." },
      { stage: "Load Balancing", reason: "Configures Istio-based load balancing with session affinity for optimal request distribution." },
      { stage: "Monitoring & Observability", reason: "Integrates Prometheus metrics and Grafana dashboards for real-time inference performance tracking." }
    ],
    citation: "Recommended based on Red Hat AI validated demos – Enterprise LLM Serving with llm-d (fictitious)."
  }
};

const pipelineConfigs = {
  rag: [
    {
      icon: FileText,
      title: "Document Upload",
      description: "Upload grounding documents (PDF, Word, text files) for knowledge base",
      completed: true,
    },
    {
      icon: Scissors,
      title: "Document Chunking",
      description: "Split documents into optimal chunk sizes with configurable overlap",
      completed: true,
      badges: ["Chunk size: 512 tokens"],
    },
    {
      icon: Grid3x3,
      title: "Embedding Generation",
      description: "Convert chunks to vector embeddings using selected embedding model",
      completed: false,
      running: true,
      badges: ["slate-30m-english-rtrvr"],
    },
    {
      icon: Database,
      title: "Vector Store",
      description: "Store vectorized content in database (Chroma, Milvus, or Elasticsearch)",
      completed: false,
      badges: ["Milvus"],
    },
    {
      icon: Search,
      title: "Retrieval Testing",
      description: "Test retrieval accuracy with sample queries and ground truth",
      completed: false,
    },
    {
      icon: Trophy,
      title: "RAG Pattern Ranking",
      description: "Evaluate and rank RAG patterns using metrics: answer_correctness, faithfulness, context_recall",
      completed: false,
    },
  ],
  finetuning: [
    {
      icon: Binary,
      title: "Data processing",
      description: "Simplifies document processing and parsing into AI-readable data for model customization and RAG applications",
      completed: true,
      badges: ["Tool: Data Prep Kit (DPK)", "Format: JSONL"],
    },
    {
      icon: Network,
      title: "Synthetic data Generation hub",
      description: "Generate high-quality data, with dynamic parameters, run-time visibility, and multilingual support",
      completed: true,
      badges: ["Generator: InstructLab (ilab)", "Method: Taxonomy-guided", "Output: 10,000 Q&A pairs"],
    },
    {
      icon: Laptop,
      title: "Training hub",
      description: "An algorithm-focused interface for common llm training, continual learning, and reinforcement learning techniques",
      completed: true,
      badges: ["Method: LoRA", "Base Model: granite-7b-lab", "Rank: r=8, Alpha: 16", "Epochs: 3"],
    },
    {
      icon: BarChart3,
      title: "Evaluations",
      description: "Simplifies the distributed execution of Evaluation jobs from popular eval frameworks or tasks",
      completed: true,
      badges: ["Framework: RAGAS + MT-Bench", "Metrics: BLEU, ROUGE-L, F1"],
    },
  ],
  synthetic: [
    {
      icon: Binary,
      title: "Data preprocessing",
      description: "Load seed examples and prepare taxonomy structure",
      completed: true,
      badges: ["Seed examples: 50 samples", "Categories: 5 classes"],
    },
    {
      icon: Network,
      title: "Synthetic data Generation hub",
      description: "Generate high-quality synthetic training data using InstructLab",
      completed: false,
      running: true,
      badges: ["Output: 5,000 samples", "Diversity score: 0.85", "Multilingual: English, Spanish, French", "Quality threshold: >0.7"],
    },
  ],
  llmserving: [
    {
      icon: Cpu,
      title: "Model Configuration",
      description: "Configure Llama-3-70B model with tensor parallelism and quantization settings",
      completed: true,
      badges: ["Model: Llama-3-70B", "Tensor Parallel: 4 GPUs", "Quantization: FP16"],
    },
    {
      icon: Server,
      title: "llm-d Distributed Setup",
      description: "Deploy llm-d with disaggregated prefill/decode pools for high-throughput serving",
      completed: true,
      badges: ["llm-d v0.1", "Prefill workers: 2", "Decode workers: 4", "vLLM backend"],
    },
    {
      icon: Zap,
      title: "KServe Integration",
      description: "Configure KServe InferenceService with OpenShift AI-native autoscaling",
      completed: false,
      running: true,
      badges: ["KServe v0.12", "REST + gRPC", "Min replicas: 2", "Max replicas: 8"],
    },
    {
      icon: Database,
      title: "Prefix Cache Config",
      description: "Enable prefix caching to optimize latency for repeated prompt patterns",
      completed: false,
      badges: ["Cache size: 32GB", "Block size: 16", "Eviction: LRU"],
    },
    {
      icon: Shield,
      title: "Load Balancing",
      description: "Configure Istio service mesh for intelligent request routing",
      completed: false,
      badges: ["Istio Gateway", "Session affinity", "Health checks"],
    },
    {
      icon: Activity,
      title: "Monitoring & Observability",
      description: "Integrate Prometheus metrics and Grafana dashboards for inference monitoring",
      completed: false,
      badges: ["Prometheus", "Grafana", "Token/s metrics", "Latency P99"],
    },
  ],
};

const leaderboardData = [
  { pattern: "Pattern 3", answerCorrectness: 0.7917, faithfulness: 0.7200, contextRecall: 0.8333, rank: 1 },
  { pattern: "Pattern 1", answerCorrectness: 0.7292, faithfulness: 0.6800, contextRecall: 0.7500, rank: 2 },
  { pattern: "Pattern 2", answerCorrectness: 0.6459, faithfulness: 0.6000, contextRecall: 0.6900, rank: 3 },
];

const accentColors = {
  rag: "border-blue-500 bg-blue-500",
  finetuning: "border-primary bg-primary",
  synthetic: "border-green-500 bg-green-500",
  llmserving: "border-purple-500 bg-purple-500",
};

interface VisualPipelineProps {
  pipelineType: PipelineType;
}

export const VisualPipeline = ({ pipelineType }: VisualPipelineProps) => {
  const pipelineStages = pipelineConfigs[pipelineType];
  const accentColor = accentColors[pipelineType];
  const explanation = pipelineExplanations[pipelineType];

  return (
    <div className="h-full flex flex-col bg-background">
      <div className="p-4 border-b border-border flex items-center justify-between bg-card">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold">Pipeline Visualization</h3>
          <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
            Pipeline Status: Generated ✓
          </span>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <Edit className="w-4 h-4" />
            Edit Pipeline
          </Button>
          <Button size="sm" className="gap-2 bg-primary hover:bg-primary/90">
            <Play className="w-4 h-4" />
            Run Pipeline
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Calendar className="w-4 h-4" />
            Schedule
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-8">
        <div className="min-w-[900px] max-w-6xl mx-auto">
          {/* Why this pipeline? Panel */}
          <div className="mb-8 p-5 bg-gradient-to-r from-sky-50 to-indigo-50 dark:from-sky-950/30 dark:to-indigo-950/30 border border-sky-200 dark:border-sky-800 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <HelpCircle className="w-5 h-5 text-sky-600 dark:text-sky-400" />
              <h4 className="font-semibold text-sky-800 dark:text-sky-300">Why this pipeline?</h4>
            </div>
            <p className="text-sm text-foreground mb-4">{explanation.summary}</p>
            
            <div className="space-y-2 mb-4">
              {explanation.stages.map((stageExp, idx) => (
                <div key={idx} className="flex items-start gap-2 text-sm">
                  <span className="font-medium text-sky-700 dark:text-sky-400 min-w-[140px] shrink-0">{stageExp.stage}:</span>
                  <span className="text-muted-foreground">{stageExp.reason}</span>
                </div>
              ))}
            </div>
            
            <div className="flex items-center gap-2 pt-3 border-t border-sky-200 dark:border-sky-800">
              <BookOpen className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground italic">{explanation.citation}</span>
            </div>
          </div>

          <div className="relative">
            {/* Pipeline stages */}
            <div className={`grid gap-8 mb-8 ${pipelineType === "rag" ? "grid-cols-3" : pipelineType === "synthetic" ? "grid-cols-2" : pipelineType === "llmserving" ? "grid-cols-3" : "grid-cols-4"}`}>
              {pipelineStages.map((stage, index) => (
                <div key={index} className="relative">
                  {/* Connection line */}
                  {index < pipelineStages.length - 1 && (
                    <div className={`absolute top-12 left-[calc(50%+32px)] w-[calc(100%+32px)] h-1 ${accentColor} z-0`}>
                      <div className={`absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-t-4 border-t-transparent border-b-4 border-b-transparent border-l-8 ${accentColor.replace('bg-', 'border-l-')}`}></div>
                    </div>
                  )}
                  
                  {/* Node */}
                  <div className="relative z-10 flex flex-col items-center">
                    <div className={`w-24 h-24 rounded-full bg-card border-4 ${accentColor.replace('bg-', 'border-')} flex items-center justify-center shadow-lg mb-4 relative`}>
                      <stage.icon className={`w-10 h-10 ${accentColor.replace('bg-', 'text-').replace('border-', 'text-')}`} />
                      {stage.completed && (
                        <div className="absolute -top-1 -right-1 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                          <CheckCircle2 className="w-4 h-4 text-white" />
                        </div>
                      )}
                      {stage.running && (
                        <div className="absolute -top-1 -right-1 w-6 h-6 bg-yellow-500 rounded-full flex items-center justify-center animate-pulse">
                          <div className="w-3 h-3 bg-white rounded-full" />
                        </div>
                      )}
                    </div>
                    <h4 className="font-semibold text-center mb-2 text-sm">{stage.title}</h4>
                    <p className="text-xs text-muted-foreground text-center mb-2">{stage.description}</p>
                    
                    {/* Badges */}
                    {stage.badges && stage.badges.length > 0 && (
                      <div className="flex flex-wrap gap-1 justify-center mt-2">
                        {stage.badges.map((badge, badgeIndex) => (
                          <span key={badgeIndex} className="text-xs bg-muted px-2 py-1 rounded-full">
                            {badge}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Running stage progress */}
            {pipelineType === "synthetic" && (
              <div className="mt-8 p-6 bg-card rounded-lg border border-border">
                <h4 className="font-semibold mb-4">Generation Progress</h4>
                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-2">
                    <span>Generating synthetic data...</span>
                    <span>67%</span>
                  </div>
                  <div className="w-full bg-secondary h-2 rounded-full overflow-hidden">
                    <div className="bg-green-500 h-full transition-all" style={{ width: "67%" }} />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Temperature: 0.7</label>
                    <Slider defaultValue={[0.7]} max={1} step={0.1} className="w-full" />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Diversity penalty: 0.3</label>
                    <Slider defaultValue={[0.3]} max={1} step={0.1} className="w-full" />
                  </div>
                  <div>
                    <label className="text-sm font-medium mb-2 block">Number of generations: 5000</label>
                    <Slider defaultValue={[5000]} max={10000} step={100} className="w-full" />
                  </div>
                </div>

                <Button className="mt-4 w-full bg-green-600 hover:bg-green-700">
                  Download Generated Data
                </Button>
              </div>
            )}
          </div>

          {/* Leaderboard for RAG */}
          {pipelineType === "rag" && (
            <div className="mt-12 p-6 bg-card rounded-lg border border-border">
              <h4 className="font-semibold mb-4 flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-500" />
                RAG Pattern Leaderboard
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-4 text-sm font-medium">Rank</th>
                      <th className="text-left py-2 px-4 text-sm font-medium">Pattern Name</th>
                      <th className="text-left py-2 px-4 text-sm font-medium">Answer Correctness</th>
                      <th className="text-left py-2 px-4 text-sm font-medium">Faithfulness</th>
                      <th className="text-left py-2 px-4 text-sm font-medium">Context Recall</th>
                    </tr>
                  </thead>
                  <tbody>
                    {leaderboardData.map((item) => (
                      <tr key={item.rank} className="border-b border-border hover:bg-muted/50">
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            {item.rank === 1 && <Medal className="w-5 h-5 text-yellow-500" />}
                            {item.rank === 2 && <Medal className="w-5 h-5 text-gray-400" />}
                            {item.rank === 3 && <Medal className="w-5 h-5 text-orange-600" />}
                            <span className="font-medium">{item.rank}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 font-medium">{item.pattern}</td>
                        <td className="py-3 px-4">{item.answerCorrectness.toFixed(4)}</td>
                        <td className="py-3 px-4">{item.faithfulness.toFixed(4)}</td>
                        <td className="py-3 px-4">{item.contextRecall.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pipeline details */}
          <div className="mt-12 p-6 bg-card rounded-lg border border-border">
            <h4 className="font-semibold mb-4">Pipeline Details</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Total Stages:</span>
                <span className="ml-2 font-medium">{pipelineStages.length}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Status:</span>
                <span className="ml-2 font-medium text-green-600">
                  {pipelineStages.some(s => s.running) ? "Running" : "Ready to Run"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Estimated Runtime:</span>
                <span className="ml-2 font-medium">
                  {pipelineType === "rag" ? "~60 minutes" : pipelineType === "synthetic" ? "~15 minutes" : pipelineType === "llmserving" ? "~30 minutes" : "~45 minutes"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground">Last Modified:</span>
                <span className="ml-2 font-medium">Just now</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
