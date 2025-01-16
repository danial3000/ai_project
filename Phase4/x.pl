:- dynamic bird_position/2.
:- dynamic pig_position/2.
:- dynamic rock_position/2.

grid_size(8).

set_bird_position(X, Y) :-
    retractall(bird_position(_, _)),
    assertz(bird_position(X, Y)).

% افزودن موقعیت خوک‌ها
add_pig_position(X, Y) :-
    assertz(pig_position(X, Y)).

% افزودن موقعیت سنگ‌ها
add_rock_position(X, Y) :-
    assertz(rock_position(X, Y)).

% حرکت‌های مجاز
move(0, X, Y, X2, Y) :- X2 is X - 1, X2 >= 0.
move(1, X, Y, X2, Y) :- grid_size(Size), X2 is X + 1, X2 < Size.
move(2, X, Y, X, Y2) :- Y2 is Y - 1, Y2 >= 0.
move(3, X, Y, X, Y2) :- grid_size(Size), Y2 is Y + 1, Y2 < Size.

% بررسی موقعیت مجاز
valid_position(X, Y) :-
    \+ rock_position(X, Y),
    X >= 0, Y >= 0,
    grid_size(Size),
    X < Size, Y < Size.

% بررسی برد
win([]).

% حل مسئله با مسیر کنش‌ها
solve_path_with_actions((X, Y), Actions) :-
    findall((PX, PY), pig_position(PX, PY), Pigs),
    solve((X, Y), Pigs, [], Actions).

solve(_, [], Actions, Actions).
solve((X, Y), [(PX, PY)|RemainingPigs], CurrentActions, FinalActions) :-
    path_to_pig((X, Y), (PX, PY), Path),
    append(CurrentActions, Path, NewActions),
    solve((PX, PY), RemainingPigs, NewActions, FinalActions).

% پیدا کردن مسیر به سمت خوک
path_to_pig(Start, End, Path) :-
    bfs([[Start]], End, [], PathWithDirections),
    extract_directions(PathWithDirections, Path).

% استخراج جهت‌ها از مسیر
extract_directions([_], []).
extract_directions([(X1, Y1), (X2, Y2)|Rest], [Direction|Directions]) :-
    direction((X1, Y1), (X2, Y2), Direction),
    extract_directions([(X2, Y2)|Rest], Directions).

% تعیین جهت حرکت بین دو موقعیت
direction((X1, Y1), (X2, Y2), 0) :- X2 is X1 - 1, Y2 =:= Y1.
direction((X1, Y1), (X2, Y2), 1) :- X2 is X1 + 1, Y2 =:= Y1.
direction((X1, Y1), (X2, Y2), 2) :- Y2 is Y1 - 1, X2 =:= X1.
direction((X1, Y1), (X2, Y2), 3) :- Y2 is Y1 + 1, X2 =:= X1.

% الگوریتم BFS برای پیدا کردن مسیر
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

% بارگذاری اطلاعات از فایل متنی
load_positions_from_file(FilePath) :-
    catch(
        (open(FilePath, read, Stream),
         load_positions(Stream, 0),
         close(Stream)),
        Error,
        (write('Error: '), writeln(Error))
    ).

load_positions(Stream, Row) :-
    read_line_to_string(Stream, Line),
    (   Line \= end_of_file
    ->  process_line(Line, Row, 0),
        NextRow is Row + 1,
        load_positions(Stream, NextRow)
    ;   true
    ).

% پردازش هر خط
process_line("", _, _).
process_line(Line, Row, Col) :-
    string_length(Line, Length),
    (   Col < Length
    ->  sub_string(Line, Col, 1, _, Char),
        (   Char = "B"
        ->  set_bird_position(Row, Col)
        ;   Char = "P"
        ->  add_pig_position(Row, Col)
        ;   Char = "R"
        ->  add_rock_position(Row, Col)
        ;   Char = "T" % فضای خالی
        ->  true       % هیچ عملی انجام نمی‌شود
        ;   writeln(['Unknown character:', Char, 'at row:', Row, 'col:', Col]) % هشدار در صورت وجود کاراکتر نامعتبر
        ),
        NextCol is Col + 1,
        process_line(Line, Row, NextCol)
    ;   true
    ).