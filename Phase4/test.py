import unittest

from pyswip import Prolog


class test (unittest.TestCase):
    def setUp(self):
        self.prolog = Prolog()
        self.prolog.consult("x.pl")

    def test_add_(self):
        self.prolog.query("add_pig_position(1, 0)")

        # گرفتن تمام موقعیت‌های خوک‌ها
        result = list(self.prolog.query("pig_position(X, Y)"))
        print(result)  # نمایش موقعیت‌های خوک‌ها

    def test_add(self):
        self.prolog.query("add_pig_position(1, 0)")
        print(list(self.prolog.query("pig_position(X, Y)")))
        self.prolog.query("add_pig_position(1, 6)")
        print(self.prolog.query("pig_position(X, Y)"))

    def test_valid_move(self):
        print(list(self.prolog.query(f"load_positions_from_file('simple.txt')")))
        print(list(self.prolog.query("rock_position(X, Y)")))

        # self.assertEqual()

    def test_load_positions(self):
        print(list(self.prolog.query("load_positions_from_file('simple.txt')")))
        print(list(self.prolog.query("solve_path_with_actions((0, 0), Actions)")))

        # pigs = list(self.prolog.query("pig_position(X, Y)"))
        # bird = list(self.prolog.query("bird_position(X, Y)"))
        # rocks = list(self.prolog.query("rock_position(X, Y)"))
        # print("Bird Position:", bird)
        # print("Pig Positions:", pigs)
        # print("Rock Positions:", rocks)
        #
        # print(list(self.prolog.query("findall((PX, PY), pig_position(PX, PY), Pigs)")))
        # print(list(self.prolog.query("path_to_pig((0, 0), (2, 0), Path)")))
        # print(list(self.prolog.query("solve((0, 0), [(2, 0),(2, 1), (4, 0)], [], Actions)")))

    def test_test_pro(self):
        print(list(self.prolog.query("find_bird_position('simple.txt')")))
        print(list(self.prolog.query("bird_position(X, Y)")))
