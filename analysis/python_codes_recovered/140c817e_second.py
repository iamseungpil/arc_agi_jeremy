def transform(grid: list[list[int]]) -> list[list[int]]:
    """
    For each cell that is 1 in the input grid:
        - set the whole row to 1 (only on zeros)
        - set the whole column to 1 (only on zeros)
        - set the centre to 2
        - set the four diagonal neighbours to 3 (only on zeros)
    All other cells are left unchanged.
    """
    if not grid or not grid[0]:
        return grid

    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]          # copy of the input

    # find all original 1s
    seeds = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]

    for r, c in seeds:
        # centre
        out[r][c] = 2

        # whole row
        for j in range(w):
            if out[r][j] == 0:
                out[r][j] = 1

        # whole column
        for i in range(h):
            if i != r and out[i][c] == 0:
                out[i][c] = 1

        # diagonals
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                out[nr][nc] = 3

    return out