from enum import IntFlag


class Verbose(IntFlag):
  SILENT = 0
  NORMAL = 1
  DEBUG = 2


class AttackModes(IntFlag):
  INPAINT = 1
  TRANSLATE = 2
