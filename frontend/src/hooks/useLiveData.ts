import { useEffect, useState } from 'react';

export function usePolling<T>(fetcher: () => Promise<T>, intervalMs = 5000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let isMounted = true;
    let timeout: number;

    const load = async () => {
      try {
        const result = await fetcher();
        if (isMounted) {
          setData(result);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setError(err as Error);
        }
      } finally {
        if (isMounted) {
          timeout = window.setTimeout(load, intervalMs);
        }
      }
    };

    load();

    return () => {
      isMounted = false;
      window.clearTimeout(timeout);
    };
  }, [fetcher, intervalMs]);

  return { data, error };
}
