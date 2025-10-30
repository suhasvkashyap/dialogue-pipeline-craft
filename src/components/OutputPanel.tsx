import { useState } from "react";
import { FileCode, Workflow } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { NotebookView } from "./NotebookView";
import { VisualPipeline } from "./VisualPipeline";
import { PipelineType } from "./ChatInterface";

interface OutputPanelProps {
  pipelineGenerated: boolean;
  pipelineType: PipelineType;
}

export const OutputPanel = ({ pipelineGenerated, pipelineType }: OutputPanelProps) => {
  const [activeTab, setActiveTab] = useState("notebook");

  if (!pipelineGenerated) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center max-w-md">
          <Workflow className="w-16 h-16 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Pipeline Generated Yet</h3>
          <p className="text-sm text-muted-foreground">
            Use the chat interface to describe your pipeline needs. I'll generate both a Jupyter notebook and a
            visual pipeline diagram for you.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
        <div className="border-b border-border bg-card px-4">
          <TabsList className="bg-transparent">
            <TabsTrigger value="notebook" className="gap-2">
              <FileCode className="w-4 h-4" />
              Output 1: Notebook
            </TabsTrigger>
            <TabsTrigger value="pipeline" className="gap-2">
              <Workflow className="w-4 h-4" />
              Output 2: Visual Pipeline
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="notebook" className="flex-1 m-0">
          <NotebookView pipelineType={pipelineType} />
        </TabsContent>

        <TabsContent value="pipeline" className="flex-1 m-0">
          <VisualPipeline pipelineType={pipelineType} />
        </TabsContent>
      </Tabs>
    </div>
  );
};
