def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    For every cell that contains the value 1:
        * set its entire row to 1
        * set its entire column to 1
        * set the intersection cell to 2
        * set the four diagonally adjacent cells to 3
    All other cells remain unchanged.
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])

    # Find all positions that contain 1
    seeds = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]

    # Work on a copy so we do not modify the input while iterating
    out = [row[:] for row in grid]

    for r, c in seeds:
        # Row
        for j in range(w):
            out[r][j] = 1
        # Column
        for i in range(h):
            out[i][c] = 1
        # Center
        out[r][c] = 2
        # Diagonals
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = 3

    return out