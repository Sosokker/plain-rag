class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

    def split_text(self, text: str) -> list[str]:
        # Start with the full document
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[0] if separators else ""
        splits = (
            text.split(separator) if separator else list(text)
        )  # character-level split

        chunks = []
        current_chunk = ""

        for part in splits:
            next_chunk = current_chunk + separator + part if current_chunk else part

            if len(next_chunk) <= self.chunk_size:
                current_chunk = next_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If any chunk is too long and we have more separators to try
        if (
            any(len(chunk) > self.chunk_size for chunk in chunks)
            and len(separators) > 1
        ):
            refined_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    refined_chunks.extend(self._split_recursive(chunk, separators[1:]))
                else:
                    refined_chunks.append(chunk)
            chunks = refined_chunks

        # Add overlap
        return self._add_overlap(chunks)

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            start = max(0, i - 1)
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                overlap_text = (
                    chunks[start][-self.chunk_overlap :]
                    if self.chunk_overlap > 0
                    else ""
                )
                merged = overlap_text + chunk
                overlapped_chunks.append(merged)
        return overlapped_chunks
