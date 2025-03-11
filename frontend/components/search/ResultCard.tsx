import { SearchResult } from '@/app/lib/api';
import { formatRelativeTime } from '@/app/lib/utils';

interface ResultCardProps {
  result: SearchResult;
}

export default function ResultCard({ result }: ResultCardProps) {
  return (
    <div className="p-4 mb-4 bg-gradient-to-b from-[#2F2F2E] to-[#2B2C2A] hover:from-[#3D3D3A] hover:to-[#393937] rounded-xl border-[0.1px] border-[#393936] transition-colors duration-100">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-medium font-serif">{result.conversation_name}</h3>
      </div>
      {/* <p className="mb-2">{result.text}</p> */}
      <div className="flex justify-between text-sm text-[#B9B5A9]">
        {/* Last message {formatRelativeTime(result.timestamp)} ago */}
        {result.text}
      </div>
    </div>
  );
}
