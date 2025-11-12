import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Server, Zap, Shield, Activity, TrendingUp, GitBranch, Database, Plus } from "lucide-react";

export default function ModelServing() {
  const deployedModels = [
    { name: "granite-7b-sentiment", version: "v1.2.0", framework: "PyTorch", endpoint: "REST", status: "Running", requests: "1.2K/h", latency: "45ms" },
    { name: "rag-embedding-model", version: "v2.0.1", framework: "TensorFlow", endpoint: "gRPC", status: "Running", requests: "890/h", latency: "32ms" },
    { name: "classification-xgboost", version: "v1.0.5", framework: "XGBoost", endpoint: "REST", status: "Scaling", requests: "2.4K/h", latency: "28ms" },
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
                <h1 className="text-3xl font-bold text-foreground">Model Serving</h1>
                <p className="text-muted-foreground mt-2">Deploy and monitor ML models at scale with real-time inference</p>
              </div>
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Plus className="w-4 h-4 mr-2" />
                Deploy Model
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Active Models</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">3</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Total Requests</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">4.5K</div>
                  <p className="text-xs text-muted-foreground mt-1">per hour</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Avg Latency</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">35ms</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Success Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">99.8%</div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Platform Capabilities</CardTitle>
                <CardDescription>Enterprise-grade model deployment inspired by KServe and SageMaker</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex gap-3">
                    <Server className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">API Endpoints</h4>
                      <p className="text-sm text-muted-foreground">Deploy models as REST & gRPC endpoints with authentication</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Database className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Multi-Framework Support</h4>
                      <p className="text-sm text-muted-foreground">PyTorch, TensorFlow, scikit-learn, XGBoost, and LLMs</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <GitBranch className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Model Registry</h4>
                      <p className="text-sm text-muted-foreground">Version control with metadata, tags, and approval workflows</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Zap className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Auto-Scaling</h4>
                      <p className="text-sm text-muted-foreground">Hardware acceleration (CPU, GPU, FPGA) with elastic scaling</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Activity className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Live Monitoring</h4>
                      <p className="text-sm text-muted-foreground">Performance, latency, drift detection, and custom metrics</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <TrendingUp className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Deployment Strategies</h4>
                      <p className="text-sm text-muted-foreground">Canary, blue-green, shadow deployment, and rollback</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Shield className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Security & Access</h4>
                      <p className="text-sm text-muted-foreground">Role and token-based authentication with audit logging</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Deployed Models</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {deployedModels.map((model, idx) => (
                    <div key={idx} className="flex items-center justify-between p-4 border border-border rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold">{model.name}</h3>
                          <Badge variant={model.status === "Running" ? "default" : "secondary"}>
                            {model.status}
                          </Badge>
                          <Badge variant="outline">{model.version}</Badge>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                          <span>{model.framework}</span>
                          <span>{model.endpoint}</span>
                          <span className="flex items-center gap-1">
                            <Activity className="w-3 h-3" />
                            {model.requests}
                          </span>
                          <span>~{model.latency}</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">Metrics</Button>
                        <Button variant="outline" size="sm">Configure</Button>
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
