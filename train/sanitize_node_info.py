import sys

for line in sys.stdin.readlines():
    new_line = line.lstrip('Nodes=cn-')
    if new_line[0] != '[':
        sys.stdout.writelines(line)
        continue

    node_from = ''
    node_to = ''
    is_from = True
    cnt = 1
    for char in new_line[1:]:
        cnt += 1
        if char == '-':
            is_from = False
            continue

        if char == ']':
            break

        if is_from:
            node_from += char
        else:
            node_to += char

    diff = int(node_to) - int(node_from)
    nodes = [node_from]
    for i in range(diff):
        nodes.append(f"{(int(node_from) + i + 1):03d}")

    sys.stdout.writelines([f'Nodes=cn-{node}{new_line[cnt:]}' for node in nodes])

