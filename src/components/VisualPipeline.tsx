import { Binary, Network, Laptop, BarChart3, CheckCircle2, Edit, Play, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";

const pipelineStages = [
  {
    icon: Binary,
    title: "Data processing",
    description:
      "Simplifies document processing and parsing into AI-readable data for model customization and RAG applications",
    completed: true,
  },
  {
    icon: Network,
    title: "Synthetic data Generation hub",
    description:
      "Generate high-quality data, with dynamic parameters, run-time visibility, and multilingual support",
    completed: true,
  },
  {
    icon: Laptop,
    title: "Training hub",
    description:
      "An algorithm-focused interface for common llm training, continual learning, and reinforcement learning techniques",
    completed: true,
  },
  {
    icon: BarChart3,
    title: "Evaluations",
    description:
      "Simplifies the distributed execution of Evaluation jobs from popular eval frameworks or tasks",
    completed: true,
  },
];

export const VisualPipeline = () => {
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

      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-6xl mx-auto">
          <div className="relative">
            {/* Pipeline stages */}
            <div className="grid grid-cols-4 gap-8 mb-8">
              {pipelineStages.map((stage, index) => (
                <div key={index} className="relative">
                  {/* Connection line */}
                  {index < pipelineStages.length - 1 && (
                    <div className="absolute top-12 left-[calc(50%+32px)] w-[calc(100%+32px)] h-1 bg-primary z-0">
                      <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0 border-t-4 border-t-transparent border-b-4 border-b-transparent border-l-8 border-l-primary"></div>
                    </div>
                  )}
                  
                  {/* Node */}
                  <div className="relative z-10 flex flex-col items-center">
                    <div className="w-24 h-24 rounded-full bg-card border-4 border-primary flex items-center justify-center shadow-lg mb-4 relative">
                      <stage.icon className="w-10 h-10 text-primary" />
                      {stage.completed && (
                        <div className="absolute -top-1 -right-1 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                          <CheckCircle2 className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </div>
                    <h4 className="font-semibold text-center mb-2 text-sm">{stage.title}</h4>
                    <p className="text-xs text-muted-foreground text-center">{stage.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline details */}
          <div className="mt-12 p-6 bg-card rounded-lg border border-border">
            <h4 className="font-semibold mb-4">Pipeline Details</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Total Stages:</span>
                <span className="ml-2 font-medium">4</span>
              </div>
              <div>
                <span className="text-muted-foreground">Status:</span>
                <span className="ml-2 font-medium text-green-600">Ready to Run</span>
              </div>
              <div>
                <span className="text-muted-foreground">Estimated Runtime:</span>
                <span className="ml-2 font-medium">~45 minutes</span>
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
