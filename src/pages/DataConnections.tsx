import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Database, Cloud, HardDrive, Lock, Activity, GitBranch, Search, Plus } from "lucide-react";

export default function DataConnections() {
  const connections = [
    { name: "PostgreSQL Production DB", type: "Database", status: "Connected", lastSync: "5m ago", health: "healthy" },
    { name: "S3 Training Data Bucket", type: "Object Storage", status: "Connected", lastSync: "1h ago", health: "healthy" },
    { name: "Milvus Vector Store", type: "Vector DB", status: "Syncing", lastSync: "Syncing", health: "syncing" },
    { name: "Elasticsearch Logs", type: "Search Engine", status: "Error", lastSync: "2d ago", health: "error" },
  ];

  const connectorTypes = [
    { name: "PostgreSQL", icon: Database, category: "Relational Database" },
    { name: "Amazon S3", icon: Cloud, category: "Object Storage" },
    { name: "Milvus", icon: Database, category: "Vector Database" },
    { name: "Snowflake", icon: HardDrive, category: "Data Warehouse" },
    { name: "MongoDB", icon: Database, category: "NoSQL Database" },
    { name: "Elasticsearch", icon: Search, category: "Search Engine" },
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
                <h1 className="text-3xl font-bold text-foreground">Data Connections</h1>
                <p className="text-muted-foreground mt-2">Seamlessly connect to enterprise, cloud, and open data sources</p>
              </div>
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Plus className="w-4 h-4 mr-2" />
                Add Connection
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Active Connections</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">4</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Healthy</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">2</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Syncing</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-secondary">1</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Issues</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-destructive">1</div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Key Features</CardTitle>
                <CardDescription>Enterprise data connectivity inspired by SageMaker and Databricks</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex gap-3">
                    <Database className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Connector Marketplace</h4>
                      <p className="text-sm text-muted-foreground">Pre-built connectors for databases, object stores, data lakes</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Lock className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Secure Credential Vault</h4>
                      <p className="text-sm text-muted-foreground">Encrypted secret management with role-based access</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Activity className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Connection Monitoring</h4>
                      <p className="text-sm text-muted-foreground">Health checks, quick tests, and performance metrics</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Cloud className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Flexible Access Patterns</h4>
                      <p className="text-sm text-muted-foreground">Batch ingestion, streaming, and direct query support</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Search className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Data Catalog Integration</h4>
                      <p className="text-sm text-muted-foreground">Discover and preview datasets with metadata search</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <GitBranch className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Lineage Tracking</h4>
                      <p className="text-sm text-muted-foreground">Monitor data provenance across pipelines and models</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Active Connections</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {connections.map((conn, idx) => (
                    <div key={idx} className="flex items-center justify-between p-4 border border-border rounded-lg">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold">{conn.name}</h3>
                          <Badge variant={
                            conn.health === "healthy" ? "default" : 
                            conn.health === "syncing" ? "secondary" : 
                            "destructive"
                          }>
                            {conn.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                          <span>{conn.type}</span>
                          <span className="flex items-center gap-1">
                            <Activity className="w-3 h-3" />
                            Last sync: {conn.lastSync}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">Test</Button>
                        <Button variant="outline" size="sm">Configure</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Available Connectors</CardTitle>
                <CardDescription>Browse and add new data source connections</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {connectorTypes.map((connector, idx) => (
                    <div key={idx} className="p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex items-center gap-3 mb-2">
                        <connector.icon className="w-5 h-5 text-primary" />
                        <h3 className="font-semibold">{connector.name}</h3>
                      </div>
                      <p className="text-sm text-muted-foreground mb-3">{connector.category}</p>
                      <Button size="sm" variant="outline">Add Connection</Button>
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
