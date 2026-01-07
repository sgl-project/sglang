import { AppShell } from './components/layout/AppShell';
import { TopBar } from './components/layout/TopBar';
import { ChatView } from './components/views/ChatView';
import { InspectView } from './components/views/InspectView';
import { ManifoldView } from './components/views/ManifoldView';
import { RouterView } from './components/views/RouterView';
import { ComparisonView } from './components/comparison';
import { InsightPanel } from './components/lens/InsightPanel';
import { TokenLensDrawer } from './components/lens/TokenLensDrawer';
import { useUIStore } from './stores/useUIStore';

export default function App() {
  const view = useUIStore((state) => state.view);

  const renderView = () => {
    switch (view) {
      case 'chat':
        return <ChatView />;
      case 'inspect':
        return <InspectView />;
      case 'manifold':
        return <ManifoldView />;
      case 'router':
        return <RouterView />;
      case 'compare':
        return <ComparisonView />;
      default:
        return <ChatView />;
    }
  };

  return (
    <AppShell>
      <TopBar />
      <div className="main-grid">
        <div className="main-content">
          {renderView()}
        </div>
        <div className="sidebar">
          <InsightPanel />
        </div>
      </div>
      {/* Token Lens Drawer - slides in from right on token hover/click */}
      <TokenLensDrawer />
    </AppShell>
  );
}
