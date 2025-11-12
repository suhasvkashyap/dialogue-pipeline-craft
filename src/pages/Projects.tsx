import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FolderOpen, Users, GitBranch, Clock, TrendingUp, Plus, Upload, Download } from "lucide-react";

export default function Projects() {
  const projects = [
    { name: "Sentiment Analysis Pipeline", status: "Active", users: 5, lastModified: "2 hours ago", workflows: 3 },
    { name: "RAG Document Processing", status: "Training", users: 3, lastModified: "1 day ago", workflows: 2 },
    { name: "Customer Churn Prediction", status: "Deployed", users: 8, lastModified: "3 days ago", workflows: 5 },
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
                <h1 className="text-3xl font-bold text-foreground">Projects</h1>
                <p className="text-muted-foreground mt-2">Organize all AI, ML, and data science initiatives in one central location</p>
              </div>
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Plus className="w-4 h-4 mr-2" />
                New Project
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Active Projects</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">{projects.length}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Total Workflows</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">10</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Collaborators</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-4xl font-bold text-primary">16</div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Key Features</CardTitle>
                <CardDescription>Comprehensive project management for AI/ML initiatives</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex gap-3">
                    <FolderOpen className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Project Dashboard</h4>
                      <p className="text-sm text-muted-foreground">Quick status overview of workflows, models, and resource usage</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Users className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Role-Based Access</h4>
                      <p className="text-sm text-muted-foreground">Admin, contributor, and viewer roles for team collaboration</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <GitBranch className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Git Integration</h4>
                      <p className="text-sm text-muted-foreground">Version control with Gitea or GitHub for experiments and code</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <TrendingUp className="w-5 h-5 text-primary shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold">Lifecycle Management</h4>
                      <p className="text-sm text-muted-foreground">Create, archive, clone, and configure project settings</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Your Projects</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {projects.map((project, idx) => (
                    <div key={idx} className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <h3 className="font-semibold">{project.name}</h3>
                          <Badge variant={project.status === "Active" ? "default" : project.status === "Training" ? "secondary" : "outline"}>
                            {project.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Users className="w-3 h-3" />
                            {project.users} users
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {project.lastModified}
                          </span>
                          <span>{project.workflows} workflows</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm">
                          <Upload className="w-4 h-4 mr-2" />
                          Export
                        </Button>
                        <Button variant="outline" size="sm">Open</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="flex gap-4">
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Import Template
              </Button>
              <Button variant="outline">
                <Download className="w-4 h-4 mr-2" />
                Export Configuration
              </Button>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
