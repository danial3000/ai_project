%% Part 1: loading the positions from file

% Read map from file
read_map(FilePath) :-
    catch(
        (open(FilePath, read, Stream),
         extract_lines(Stream, 0),
         close(Stream)),
        Error,
        (write('Error: '), writeln(Error))
    ).

extract_lines(Stream, Row) :-
    read_line_to_string(Stream, Line),
    (   Line \= end_of_file
    ->  extract_cells(Line, Row, 0),
        NextRow is Row + 1,
        extract_lines(Stream, NextRow)
    ;   true
    ).

% process each line
extract_cells("", _, _).
extract_cells(Line, Row, Col) :-
    string_length(Line, Length),
    (   Col < Length
    ->  sub_string(Line, Col, 1, _, Status),
        (   Status = "B"
        ->  set_bird_position(Row, Col)
        ;   Status = "P"
        ->  add_pig_position(Row, Col)
        ;   Status = "R"
        ->  add_rock_position(Row, Col)
        ;   Status = "T" % Empty space
        ->  true       % Nothing done
        ;   writeln(['Unknown character:', Status, 'at row:', Row, 'col:', Col]) % Invalid character
        ),
        NextCol is Col + 1,
        extract_cells(Line, Row, NextCol)
    ;   true
    ).

%% Part 2: getting the past



:- dynamic bird_position/2.
:- dynamic pig_position/2.
:- dynamic rock_position/2.

grid_size(8).

set_bird_position(X, Y) :-
    retractall(bird_position(_, _)),
    assertz(bird_position(X, Y)).

% Add pigs position
add_pig_position(X, Y) :-
    assertz(pig_position(X, Y)).

% Add rocks position
add_rock_position(X, Y) :-
    assertz(rock_position(X, Y)).

% Available moves
move(0, X, Y, X2, Y) :- X2 is X - 1, X2 >= 0.
move(1, X, Y, X2, Y) :- grid_size(Size), X2 is X + 1, X2 < Size.
move(2, X, Y, X, Y2) :- Y2 is Y - 1, Y2 >= 0.
move(3, X, Y, X, Y2) :- grid_size(Size), Y2 is Y + 1, Y2 < Size.

% Valid moves
valid_position(X, Y) :-
    \+ rock_position(X, Y),
    X >= 0, Y >= 0,
    grid_size(Size),
    X < Size, Y < Size.

% Win check
win([]).

% Solving problem with actions
solve_path_with_actions((X, Y), Actions) :-
    findall((PX, PY), pig_position(PX, PY), Pigs),
    solve((X, Y), Pigs, [], Actions).

solve(_, [], Actions, Actions).
solve((X, Y), [(PX, PY)|RemainingPigs], CurrentActions, FinalActions) :-
    path_to_pig((X, Y), (PX, PY), Path),
    append(CurrentActions, Path, NewActions),
    solve((PX, PY), RemainingPigs, NewActions, FinalActions).

% Finding path to pig
path_to_pig(Start, End, Path) :-
    bfs([[Start]], End, [], PathWithDirections),
    extract_directions(PathWithDirections, Path).

% Extract directions to pig
extract_directions([_], []).
extract_directions([(X1, Y1), (X2, Y2)|Rest], [Direction|Directions]) :-
    direction((X1, Y1), (X2, Y2), Direction),
    extract_directions([(X2, Y2)|Rest], Directions).

% Determining the direction from the position X to Y
direction((X1, Y1), (X2, Y2), 0) :- X2 is X1 - 1, Y2 =:= Y1.
direction((X1, Y1), (X2, Y2), 1) :- X2 is X1 + 1, Y2 =:= Y1.
direction((X1, Y1), (X2, Y2), 2) :- Y2 is Y1 - 1, X2 =:= X1.
direction((X1, Y1), (X2, Y2), 3) :- Y2 is Y1 + 1, X2 =:= X1.

% BFS algorithm to find the route
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
