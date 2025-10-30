import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { ChatInterface, PipelineType } from "@/components/ChatInterface";
import { OutputPanel } from "@/components/OutputPanel";

const Index = () => {
  const [pipelineGenerated, setPipelineGenerated] = useState(false);
  const [pipelineType, setPipelineType] = useState<PipelineType>("finetuning");

  const handlePipelineGenerated = (type: PipelineType) => {
    setPipelineType(type);
    setPipelineGenerated(true);
  };

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <div className="flex-1 flex overflow-hidden">
          <div className="w-[40%] border-r border-border">
            <ChatInterface onPipelineGenerated={handlePipelineGenerated} />
          </div>
          
          <div className="flex-1">
            <OutputPanel pipelineGenerated={pipelineGenerated} pipelineType={pipelineType} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
