import { Sidebar } from "@/components/Sidebar";
import { Header } from "@/components/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { User, Shield, Zap, Bell, Monitor, Key, FileText, Eye } from "lucide-react";

export default function Settings() {
  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-7xl mx-auto space-y-8">
            <div>
              <h1 className="text-3xl font-bold text-foreground">Settings</h1>
              <p className="text-muted-foreground mt-2">Configure workspace, security, and platform preferences</p>
            </div>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <User className="w-5 h-5 text-primary" />
                  <CardTitle>User Profile</CardTitle>
                </div>
                <CardDescription>Manage your account settings and preferences</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label>Display Name</Label>
                    <p className="text-sm text-muted-foreground mt-1">John Doe</p>
                  </div>
                  <div>
                    <Label>Email</Label>
                    <p className="text-sm text-muted-foreground mt-1">john.doe@company.com</p>
                  </div>
                  <div>
                    <Label>Role</Label>
                    <p className="text-sm text-muted-foreground mt-1">Administrator</p>
                  </div>
                  <div>
                    <Label>Team</Label>
                    <p className="text-sm text-muted-foreground mt-1">Data Science</p>
                  </div>
                </div>
                <Button variant="outline">Edit Profile</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Shield className="w-5 h-5 text-primary" />
                  <CardTitle>Security & Authentication</CardTitle>
                </div>
                <CardDescription>SSO integration and access control</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Two-Factor Authentication</Label>
                    <p className="text-sm text-muted-foreground">Add an extra layer of security</p>
                  </div>
                  <Switch />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>SSO Integration</Label>
                    <p className="text-sm text-muted-foreground">Enable single sign-on with corporate identity provider</p>
                  </div>
                  <Button variant="outline" size="sm">Configure</Button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Session Timeout</Label>
                    <p className="text-sm text-muted-foreground">Auto-logout after 30 minutes of inactivity</p>
                  </div>
                  <Button variant="outline" size="sm">Change</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Zap className="w-5 h-5 text-primary" />
                  <CardTitle>Resource Quotas</CardTitle>
                </div>
                <CardDescription>Compute, storage, and network limits</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div>
                    <Label>CPU Quota</Label>
                    <p className="text-2xl font-bold text-primary mt-2">32 vCPUs</p>
                    <p className="text-xs text-muted-foreground">16 used / 50%</p>
                  </div>
                  <div>
                    <Label>GPU Quota</Label>
                    <p className="text-2xl font-bold text-primary mt-2">4 GPUs</p>
                    <p className="text-xs text-muted-foreground">2 used / 50%</p>
                  </div>
                  <div>
                    <Label>Storage</Label>
                    <p className="text-2xl font-bold text-primary mt-2">500 GB</p>
                    <p className="text-xs text-muted-foreground">287 GB used / 57%</p>
                  </div>
                </div>
                <Button variant="outline">Request Quota Increase</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Eye className="w-5 h-5 text-primary" />
                  <CardTitle>UI Preferences</CardTitle>
                </div>
                <CardDescription>Customize dashboard layout and accessibility</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Dark Mode</Label>
                    <p className="text-sm text-muted-foreground">Use dark color scheme</p>
                  </div>
                  <Switch />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Compact View</Label>
                    <p className="text-sm text-muted-foreground">Reduce spacing for more content</p>
                  </div>
                  <Switch />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Experimental Features</Label>
                    <p className="text-sm text-muted-foreground">Enable beta functionality and new capabilities</p>
                  </div>
                  <Switch />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Bell className="w-5 h-5 text-primary" />
                  <CardTitle>Notifications & Monitoring</CardTitle>
                </div>
                <CardDescription>Alert configuration and observability integrations</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">Receive alerts for pipeline failures and deployments</p>
                  </div>
                  <Switch defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Slack Integration</Label>
                    <p className="text-sm text-muted-foreground">Send notifications to Slack channels</p>
                  </div>
                  <Button variant="outline" size="sm">Connect</Button>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Prometheus/Grafana</Label>
                    <p className="text-sm text-muted-foreground">Export metrics to observability tools</p>
                  </div>
                  <Button variant="outline" size="sm">Configure</Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Key className="w-5 h-5 text-primary" />
                  <CardTitle>API Access</CardTitle>
                </div>
                <CardDescription>Manage API keys and tokens</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 border border-border rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Production API Key</Label>
                      <p className="text-sm text-muted-foreground font-mono">oai-prod-••••••••••••3f2a</p>
                      <p className="text-xs text-muted-foreground mt-1">Created 3 months ago</p>
                    </div>
                    <Button variant="outline" size="sm">Revoke</Button>
                  </div>
                </div>
                <Button variant="outline">Generate New Key</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <Monitor className="w-5 h-5 text-primary" />
                  <CardTitle>Platform Information</CardTitle>
                </div>
                <CardDescription>Version and documentation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Label>Platform Version</Label>
                    <p className="text-sm text-muted-foreground mt-1">OpenShift AI v2.5.0</p>
                  </div>
                  <div>
                    <Label>Last Updated</Label>
                    <p className="text-sm text-muted-foreground mt-1">January 15, 2025</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Button variant="outline">
                    <FileText className="w-4 h-4 mr-2" />
                    Documentation
                  </Button>
                  <Button variant="outline">View Audit Logs</Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
