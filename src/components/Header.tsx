import { Bell, User, ChevronDown } from "lucide-react";

export const Header = () => {
  return (
    <header className="h-16 bg-card border-b border-border flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <div className="text-sm text-muted-foreground">
          Projects &gt; Chat with Lightspeed &gt; AI Pipeline Builder
        </div>
      </div>
      
      <div className="flex items-center gap-4">
        <button className="relative p-2 hover:bg-muted rounded-md transition-colors">
          <Bell className="w-5 h-5" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-primary rounded-full"></span>
        </button>
        
        <button className="flex items-center gap-2 px-3 py-2 hover:bg-muted rounded-md transition-colors">
          <div className="w-8 h-8 bg-secondary rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-secondary-foreground" />
          </div>
          <span className="text-sm font-medium">Admin User</span>
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
