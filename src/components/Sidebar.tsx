import { LayoutDashboard, GitBranch, Server, Sparkles, Database, Settings } from "lucide-react";
import { Link, useLocation } from "react-router-dom";

const menuItems = [
  { icon: LayoutDashboard, label: "Projects", path: "/projects" },
  { icon: GitBranch, label: "Data Science Pipelines", path: "/pipelines" },
  { icon: Server, label: "Model Serving", path: "/model-serving" },
  { icon: Sparkles, label: "Model Customization", path: "/" },
  { icon: Database, label: "Data Connections", path: "/data-connections" },
  { icon: Settings, label: "Settings", path: "/settings" },
];

export const Sidebar = () => {
  const location = useLocation();
  
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
        {menuItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.label}
              to={item.path}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md mb-1 transition-colors ${
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "text-foreground hover:bg-muted"
              }`}
            >
              <item.icon className="w-4 h-4" />
              <span className="text-sm font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
};
