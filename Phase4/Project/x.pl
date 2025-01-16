% Map loading and parsing remains the same as before...
read_map(File) :-
    catch(
        (open(File, read, Stream),
         read_grid(Stream, 0),
         close(Stream)),
        Error,
        (write('Error: '), writeln(Error))
    ).

read_grid(Stream, Row) :-
    read_line_to_string(Stream, Line),
    (   Line \= end_of_file
    ->  parse_row(Line, Row, 0),
        NextRow is Row + 1,
        read_grid(Stream, NextRow)
    ;   true
    ).

parse_row("", _, _).
parse_row(Line, Row, Col) :-
    string_length(Line, Len),
    (   Col < Len
    ->  sub_string(Line, Col, 1, _, Cell),
        process_cell(Cell, Row, Col),
        NextCol is Col + 1,
        parse_row(Line, Row, NextCol)
    ;   true
    ).

process_cell("B", Row, Col) :- set_bird(Row, Col).
process_cell("P", Row, Col) :- add_pig(Row, Col).
process_cell("R", Row, Col) :- add_rock(Row, Col).
process_cell("T", _, _).     % Empty cell
process_cell(Unknown, Row, Col) :-
    writeln(['Invalid cell:', Unknown, 'at', Row, Col]).

% Dynamic predicates for game state
:- dynamic bird_pos/2.
:- dynamic pig_pos/2.
:- dynamic rock_pos/2.

grid_size(8).

% State management
set_bird(X, Y) :-
    retractall(bird_pos(_, _)),
    assertz(bird_pos(X, Y)).

add_pig(X, Y) :-
    assertz(pig_pos(X, Y)).

add_rock(X, Y) :-
    assertz(rock_pos(X, Y)).

% Movement definition
move(up,    X, Y, X2, Y) :- X2 is X - 1, X2 >= 0.
move(down,  X, Y, X2, Y) :- grid_size(S), X2 is X + 1, X2 < S.
move(left,  X, Y, X, Y2) :- Y2 is Y - 1, Y2 >= 0.
move(right, X, Y, X, Y2) :- grid_size(S), Y2 is Y + 1, Y2 < S.

dir_to_num(up, 0).
dir_to_num(down, 1).
dir_to_num(left, 2).
dir_to_num(right, 3).

% Position validation
valid_pos(X, Y) :-
    \+ rock_pos(X, Y),
    grid_size(S),
    X >= 0, Y >= 0,
    X < S, Y < S.

% Main solving predicates
find_path :-
    bird_pos(BX, BY),
    find_path_to_pigs((BX, BY), Path),
    convert_path_to_actions(Path, Actions),
    writeln(Actions).

find_path_to_pigs(Start, Path) :-
    findall((PX, PY), pig_pos(PX, PY), Pigs),
    find_path_through_pigs(Start, Pigs, [], Path).

find_path_through_pigs(_, [], Path, Path).
find_path_through_pigs(Start, [Pig|Pigs], CurrentPath, FinalPath) :-
    astar(Start, Pig, PigPath),  % Using A* instead of BFS
    append(CurrentPath, PigPath, NewPath),
    find_path_through_pigs(Pig, Pigs, NewPath, FinalPath).

% A* implementation
astar(Start, Goal, Path) :-
    manhattan_distance(Start, Goal, H),
    astar_search([node(Start, [], 0, H)], Goal, [], Path).

astar_search([node(Goal, Path, _, _)|_], Goal, _, RevPath) :-
    reverse([Goal|Path], RevPath).

astar_search([node(Pos, Path, G, _)|Queue], Goal, Visited, FinalPath) :-
    findall(node(NextPos, [Pos|Path], NewG, NewF),
            (Pos = (X, Y),
             move(_, X, Y, NX, NY),
             NextPos = (NX, NY),
             valid_pos(NX, NY),
             \+ member(NextPos, Visited),
             NewG is G + 1,
             manhattan_distance(NextPos, Goal, H),
             NewF is NewG + H),
            NewNodes),
    append(Queue, NewNodes, UnsortedQueue),
    sort_queue(UnsortedQueue, SortedQueue),
    astar_search(SortedQueue, Goal, [Pos|Visited], FinalPath).

% Uniform Cost Search implementation
ucs(Start, Goal, Path) :-
    ucs_search([node(Start, [], 0)], Goal, [], Path).

ucs_search([node(Goal, Path, _)|_], Goal, _, RevPath) :-
    reverse([Goal|Path], RevPath).

ucs_search([node(Pos, Path, Cost)|Queue], Goal, Visited, FinalPath) :-
    findall(node(NextPos, [Pos|Path], NewCost),
            (Pos = (X, Y),
             move(_, X, Y, NX, NY),
             NextPos = (NX, NY),
             valid_pos(NX, NY),
             \+ member(NextPos, Visited),
             NewCost is Cost + 1),
            NewNodes),
    append(Queue, NewNodes, UnsortedQueue),
    sort_queue_ucs(UnsortedQueue, SortedQueue),
    ucs_search(SortedQueue, Goal, [Pos|Visited], FinalPath).

% Breadth First Search implementation
bfs([[End|Visited]|_], End, _, Path) :-
    reverse([End|Visited], Path).

bfs([[Current|Visited]|Rest], End, Seen, Path) :-
    Current = (CurrentX, CurrentY),
    findall([Next, Current|Visited],
            (move(Direction, CurrentX, CurrentY, NextX, NextY),
             valid_position(NextX, NextY),
             Next = (NextX, NextY),
             \+ member(Next, Seen)),
            NextPaths),
    append(Rest, NextPaths, NewQueue),
    bfs(NewQueue, End, [Current|Seen], Path).

% Helper predicates
manhattan_distance((X1, Y1), (X2, Y2), D) :-
    D is abs(X2 - X1) + abs(Y2 - Y1).

sort_queue(Queue, SortedQueue) :-
    sort(2, @=<, Queue, SortedQueue).

sort_queue_ucs(Queue, SortedQueue) :-
    sort(2, @=<, Queue, SortedQueue).

% Convert path to action numbers
convert_path_to_actions([], []).
convert_path_to_actions([_], []).
convert_path_to_actions([(X1,Y1), (X2,Y2)|Rest], [Action|Actions]) :-
    get_action((X1,Y1), (X2,Y2), Action),
    convert_path_to_actions([(X2,Y2)|Rest], Actions).

get_action((X1,Y1), (X2,Y2), Action) :-
    (X2 =:= X1 - 1, Y2 =:= Y1 -> dir_to_num(up, Action)    ;
     X2 =:= X1 + 1, Y2 =:= Y1 -> dir_to_num(down, Action)  ;
     X2 =:= X1, Y2 =:= Y1 - 1 -> dir_to_num(left, Action)  ;
     X2 =:= X1, Y2 =:= Y1 + 1 -> dir_to_num(right, Action)).