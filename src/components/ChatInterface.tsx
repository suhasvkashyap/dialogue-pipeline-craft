import { useState } from "react";
import { Send, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const starterPrompts = [
  { 
    text: "Create a RAG system using Mistral-7B that lets our customer support agents ask natural language questions about our internal product documentation stored in Elasticsearch and returns concise, source-linked answers within 2 seconds.", 
    type: "rag" as const,
    label: "RAG: Customer Support Knowledge Base"
  },
  { 
    text: "Create a fine-tuning pipeline that adapts an open-source LLM to automatically classify incoming insurance claim emails into categories like auto, property, health, and fraud_review, and suggest a priority level (low/medium/high).", 
    type: "finetuning" as const,
    label: "Fine-Tuning: Insurance Claim Email Triage"
  },
  { 
    text: "Create a synthetic data generation pipeline that produces realistic but fully de-identified customer usage records for a telco churn prediction model, so data scientists can experiment without accessing production PII.", 
    type: "synthetic" as const,
    label: "Synthetic Data: Telco Churn Prediction"
  },
  { 
    text: "Deploy a distributed LLM inference service using llm-d on Red Hat OpenShift AI that serves Llama-3-70B across multiple GPU nodes with automatic load balancing, prefix caching for improved throughput, and KServe-compatible REST/gRPC endpoints for our enterprise chatbot application.", 
    type: "llmserving" as const,
    label: "LLM Serving: Distributed Llama-3-70B with llm-d"
  },
];

export type PipelineType = "rag" | "finetuning" | "synthetic" | "llmserving";

interface ChatInterfaceProps {
  onPipelineGenerated: (type: PipelineType) => void;
}

export const ChatInterface = ({ onPipelineGenerated }: ChatInterfaceProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm your AI pipeline assistant. I can help you create model customization pipelines. Try one of the suggestions below or describe what you'd like to build.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (text?: string, pipelineType?: PipelineType) => {
    const messageText = text || input;
    if (!messageText.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: messageText };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Determine pipeline type from message
    let detectedType: PipelineType = pipelineType || "finetuning";
    
    // Get AI response based on pipeline type
    const responses = {
      rag: "Analyzing your request... Identifying optimal pipeline configuration... I'll create a watsonx AutoAI-style RAG optimization pipeline with document chunking, embedding generation, vector storage, and pattern ranking...",
      finetuning: "Analyzing your request... I'll create a fine-tuning pipeline using LoRA for efficient training. I've selected Data Prep Kit, InstructLab, and RAGAS based on your use case...",
      synthetic: "Analyzing your request... I'll create a focused synthetic data generation workflow. Since you only need data generation, I'm showing just the preprocessing and generation stages...",
      llmserving: "Analyzing your request... I'll create a distributed LLM serving pipeline using llm-d on OpenShift AI. This includes multi-node GPU deployment, KServe integration, prefix caching optimization, and production-grade monitoring..."
    };

    // Simulate AI response
    setTimeout(() => {
      const assistantMessage: Message = {
        role: "assistant",
        content: responses[detectedType],
      };
      setMessages((prev) => [...prev, assistantMessage]);

      setTimeout(() => {
        setIsLoading(false);
        const successMessage: Message = {
          role: "assistant",
          content:
            "✓ Pipeline generated successfully! I've created both a Jupyter notebook and a visual pipeline diagram. You can see them in the Output tabs on the right.",
        };
        setMessages((prev) => [...prev, successMessage]);
        onPipelineGenerated(detectedType);
      }, 2000);
    }, 1000);
  };

  return (
    <div className="flex flex-col h-full bg-card border-r border-border">
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold text-lg">AI Pipeline Assistant</h2>
        <p className="text-sm text-muted-foreground">Describe your pipeline needs</p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
          >
            <div
              className={`max-w-[85%] rounded-lg px-4 py-3 ${
                message.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-foreground"
              }`}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start animate-fade-in">
            <div className="bg-muted rounded-lg px-4 py-3 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin-slow" />
              <span className="text-sm">Generating pipeline...</span>
            </div>
          </div>
        )}
      </div>

      {messages.length === 1 && (
        <div className="px-4 pb-4">
          <p className="text-xs text-muted-foreground mb-2">Try these examples:</p>
          <div className="flex flex-col gap-2">
            {starterPrompts.map((prompt, index) => (
              <button
                key={index}
                onClick={() => handleSend(prompt.text, prompt.type)}
                className="text-left text-sm px-3 py-2 rounded-md bg-muted hover:bg-muted/80 transition-colors"
              >
                <span className="font-medium text-primary">{prompt.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="p-4 border-t border-border">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
            placeholder="Describe your pipeline..."
            className="flex-1 px-3 py-2 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            disabled={isLoading}
          />
          <Button
            onClick={() => handleSend()}
            disabled={!input.trim() || isLoading}
            size="icon"
            className="bg-primary hover:bg-primary/90"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};
