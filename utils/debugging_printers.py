def prColour(skk, colour, end='\n'):
    if type(skk) is not str:
        skk = str(skk)
    print("\033[{}m{}\033[00m" .format(colour,skk),end=end)

def prGreen(skk,end='\n'): prColour(skk,'92',end)
def prRed(skk,end='\n'): prColour(skk,'91',end)
def prCyan(skk,end='\n'): prColour(skk,'96',end)
def prYellow(skk,end='\n'): prColour(skk,'33',end)
def prWhite(skk,end='\n'): prColour(skk,'97',end)
def prMagenta(skk,end='\n'): prColour(skk,'95',end)
def prOrange(skk,end='\n'): prColour(skk,'93',end)
def prDarkRed(skk,end='\n'): prColour(skk,'31',end)
def prDarkBlue(skk,end='\n'): prColour(skk,'34',end)
