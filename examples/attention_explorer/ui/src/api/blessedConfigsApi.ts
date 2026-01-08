/**
 * Blessed Configs API - Client for backend persistence
 *
 * Communicates with the RAPIDS sidecar's blessed configs endpoints
 * to persist approved quantization configurations.
 */

import { BlessedConfig } from '../stores/useParetoStore';

// Default sidecar URL - can be configured via environment or settings
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SIDECAR_URL = (import.meta as any).env?.VITE_SIDECAR_URL || 'http://localhost:9000';

export interface BlessedConfigsResponse {
  configs: BlessedConfig[];
}

export interface SaveConfigResponse {
  status: 'saved' | 'error';
  config_id?: string;
  error?: string;
}

export interface DeleteConfigResponse {
  status: 'deleted' | 'error';
  config_id?: string;
  error?: string;
}

/**
 * Fetch all blessed configs from the backend.
 */
export async function fetchBlessedConfigs(): Promise<BlessedConfig[]> {
  try {
    const response = await fetch(`${SIDECAR_URL}/blessed-configs`);
    if (!response.ok) {
      console.warn('Failed to fetch blessed configs:', response.statusText);
      return [];
    }
    const data: BlessedConfigsResponse = await response.json();
    return data.configs || [];
  } catch (error) {
    console.warn('Failed to fetch blessed configs (sidecar may not be running):', error);
    return [];
  }
}

/**
 * Save a blessed config to the backend.
 */
export async function saveBlessedConfig(config: BlessedConfig): Promise<boolean> {
  try {
    const response = await fetch(`${SIDECAR_URL}/blessed-configs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    if (!response.ok) {
      console.warn('Failed to save blessed config:', response.statusText);
      return false;
    }
    return true;
  } catch (error) {
    console.warn('Failed to save blessed config (sidecar may not be running):', error);
    return false;
  }
}

/**
 * Delete a blessed config from the backend.
 */
export async function deleteBlessedConfig(configId: string): Promise<boolean> {
  try {
    const response = await fetch(`${SIDECAR_URL}/blessed-configs/${configId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      console.warn('Failed to delete blessed config:', response.statusText);
      return false;
    }
    return true;
  } catch (error) {
    console.warn('Failed to delete blessed config (sidecar may not be running):', error);
    return false;
  }
}

/**
 * Sync blessed configs with backend.
 * Fetches from backend and merges with local state.
 */
export async function syncBlessedConfigs(
  localConfigs: BlessedConfig[]
): Promise<BlessedConfig[]> {
  const remoteConfigs = await fetchBlessedConfigs();

  // Merge: remote configs take precedence, but keep local-only configs
  const remoteIds = new Set(remoteConfigs.map((c) => c.id));
  const localOnly = localConfigs.filter((c) => !remoteIds.has(c.id));

  // Save local-only configs to remote
  for (const config of localOnly) {
    await saveBlessedConfig(config);
  }

  // Return merged list (remote + newly synced local)
  return [...remoteConfigs, ...localOnly];
}

/**
 * Check if the sidecar is available and has blessed configs enabled.
 */
export async function checkSidecarAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${SIDECAR_URL}/stats`);
    if (!response.ok) return false;
    const data = await response.json();
    return data.blessed_configs?.enabled !== false;
  } catch {
    return false;
  }
}
