import { LayoutDashboard, GitBranch, Server, Sparkles, Database, Settings } from "lucide-react";

const menuItems = [
  { icon: LayoutDashboard, label: "Projects", active: false },
  { icon: GitBranch, label: "Data Science Pipelines", active: false },
  { icon: Server, label: "Model Serving", active: false },
  { icon: Sparkles, label: "Model Customization", active: true },
  { icon: Database, label: "Data Connections", active: false },
  { icon: Settings, label: "Settings", active: false },
];

export const Sidebar = () => {
  return (
    <aside className="w-64 bg-card border-r border-border flex flex-col">
      <div className="p-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">OS</span>
          </div>
          <div>
            <h1 className="font-semibold text-sm">OpenShift AI</h1>
            <p className="text-xs text-muted-foreground">Model Hub</p>
          </div>
        </div>
      </div>
      
      <nav className="flex-1 px-3">
        {menuItems.map((item) => (
          <button
            key={item.label}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md mb-1 transition-colors ${
              item.active
                ? "bg-primary text-primary-foreground"
                : "text-foreground hover:bg-muted"
            }`}
          >
            <item.icon className="w-4 h-4" />
            <span className="text-sm font-medium">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
};
