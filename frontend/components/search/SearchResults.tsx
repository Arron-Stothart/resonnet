import { SearchResult } from '@/app/lib/api';
import ResultCard from './ResultCard';

interface SearchResultsProps {
  results: SearchResult[];
  isLoading: boolean;
}

export default function SearchResults({ results, isLoading }: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent-main-100)]" />
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-8 text-[var(--text-400)]">
        No results found
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {results.map((result) => (
        <ResultCard key={result.message_id} result={result} />
      ))}
    </div>
  );
}
