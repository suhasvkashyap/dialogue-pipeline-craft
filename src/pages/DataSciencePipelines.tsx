import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GitBranch, Play, Clock, Zap, Settings2, Download, TrendingUp } from "lucide-react";

export default function DataSciencePipelines() {
  const pipelines = [
    { name: "RAG Training Pipeline", runs: 45, lastRun: "Running", duration: "25m", status: "success" },
    { name: "Batch Inference Job", runs: 128, lastRun: "2h ago", duration: "15m", status: "success" },
    { name: "Data Preprocessing", runs: 67, lastRun: "Failed", duration: "8m", status: "failed" },
  ];

  const templates = [
    { name: "Fine-tuning Workflow", description: "End-to-end model fine-tuning with LoRA" },
    { name: "RAG Pipeline", description: "Document chunking, embedding, and retrieval" },
    { name: "Data Cleaning", description: "Automated data quality and transformation" },
    { name: "Model Evaluation", description: "Comprehensive model performance assessment" },
  ];

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-7xl mx-auto space-y-8">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-foreground">Data Science Pipelines</h1>
                <p className="text-muted-foreground mt-2">Build, visualize, and automate end-to-end workflows</p>
              </div>
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Play className="w-4 h-4 mr-2" />
                Create Pipeline
              </Button>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Key Capabilities</CardTitle>
                <CardDescription>Advanced pipeline orchestration inspired by KubeFlow and Vertex AI</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex gap-3">
                    <GitBranch className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Visual Pipeline Builder</h4>
                      <p className="text-sm text-muted-foreground">Drag-and-drop DAG-style workflow creation like KubeFlow/Elyra</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Download className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Pipeline Templates</h4>
                      <p className="text-sm text-muted-foreground">Pre-built workflows for training, RAG, evaluation, and batch scoring</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Settings2 className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Parameterized Pipelines</h4>
                      <p className="text-sm text-muted-foreground">Runtime configuration with dynamic parameter injection</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <TrendingUp className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Experiment Tracking</h4>
                      <p className="text-sm text-muted-foreground">Run history, lineage tracking, and artifact versioning</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Zap className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Parallel Execution</h4>
                      <p className="text-sm text-muted-foreground">Flexible scheduling with parallel and looping controls</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Clock className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Resource Analytics</h4>
                      <p className="text-sm text-muted-foreground">Cost and usage metrics per pipeline and task</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Pipeline Templates</CardTitle>
                <CardDescription>Get started quickly with pre-configured workflows</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {templates.map((template, idx) => (
                    <div key={idx} className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors">
                      <h3 className="font-semibold mb-2">{template.name}</h3>
                      <p className="text-sm text-muted-foreground mb-4">{template.description}</p>
                      <Button size="sm" variant="outline">Use Template</Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recent Pipeline Runs</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {pipelines.map((pipeline, idx) => (
                    <div key={idx} className="flex items-center justify-between p-4 border border-border rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold">{pipeline.name}</h3>
                          <Badge variant={pipeline.status === "success" ? "default" : "destructive"}>
                            {pipeline.lastRun}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                          <span>{pipeline.runs} runs</span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {pipeline.duration}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Play className="w-4 h-4 mr-2" />
                          Run
                        </Button>
                        <Button variant="outline" size="sm">View</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
