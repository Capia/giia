# Note - 1
# Only some patterns have bull and bear versions.
# However, to make the process unified and for codability purposes
# all patterns are labeled with "_Bull" and "_Bear" tags.
# Both versions of the single patterns are given same performance rank,
# since they will always return only 1 version.

# Note - 2
# The ranks were determined based on performance noted in the following websites:
# http://www.lykoff.com/market/crypt
# http://www.thepatternsite.com/rank.html
import sys

candle_rankings = {
    "CDL3LINESTRIKE_Bull": 1,
    "CDL3LINESTRIKE_Bear": 2,
    "CDL3BLACKCROWS_Bull": 3,
    "CDL3BLACKCROWS_Bear": 3,
    "CDLEVENINGSTAR_Bull": 4,
    "CDLEVENINGSTAR_Bear": 4,
    "CDLLONGLINE_Bull": 5,
    "CDLLONGLINE_Bear": 5,
    "CDLTASUKIGAP_Bull": 5,
    "CDLTASUKIGAP_Bear": 5,
    "CDLINVERTEDHAMMER_Bull": 6,
    "CDLINVERTEDHAMMER_Bear": 6,
    "CDLMATCHINGLOW_Bull": 7,
    "CDLMATCHINGLOW_Bear": 7,
    "CDLABANDONEDBABY_Bull": 8,
    "CDLABANDONEDBABY_Bear": 8,
    "CDLBREAKAWAY_Bull": 10,
    "CDLBREAKAWAY_Bear": 10,
    "CDLMORNINGSTAR_Bull": 12,
    "CDLMORNINGSTAR_Bear": 12,
    "CDLPIERCING_Bull": 13,
    "CDLPIERCING_Bear": 13,
    "CDLSTICKSANDWICH_Bull": 14,
    "CDLSTICKSANDWICH_Bear": 14,
    "CDLTHRUSTING_Bull": 15,
    "CDLTHRUSTING_Bear": 15,
    "CDLINNECK_Bull": 17,
    "CDLINNECK_Bear": 17,
    "CDL3INSIDE_Bull": 20,
    "CDL3INSIDE_Bear": 56,
    "CDLHOMINGPIGEON_Bull": 21,
    "CDLHOMINGPIGEON_Bear": 21,
    "CDLDARKCLOUDCOVER_Bull": 22,
    "CDLDARKCLOUDCOVER_Bear": 22,
    "CDLIDENTICAL3CROWS_Bull": 24,
    "CDLIDENTICAL3CROWS_Bear": 24,
    "CDLMORNINGDOJISTAR_Bull": 25,
    "CDLMORNINGDOJISTAR_Bear": 25,
    "CDLXSIDEGAP3METHODS_Bull": 27,
    "CDLXSIDEGAP3METHODS_Bear": 26,
    "CDLTRISTAR_Bull": 28,
    "CDLTRISTAR_Bear": 76,
    "CDLGAPSIDESIDEWHITE_Bull": 46,
    "CDLGAPSIDESIDEWHITE_Bear": 29,
    "CDLEVENINGDOJISTAR_Bull": 30,
    "CDLEVENINGDOJISTAR_Bear": 30,
    "CDL3WHITESOLDIERS_Bull": 32,
    "CDL3WHITESOLDIERS_Bear": 32,
    "CDLONNECK_Bull": 33,
    "CDLONNECK_Bear": 33,
    "CDL3OUTSIDE_Bull": 34,
    "CDL3OUTSIDE_Bear": 39,
    "CDLRICKSHAWMAN_Bull": 35,
    "CDLRICKSHAWMAN_Bear": 35,
    "CDLSEPARATINGLINES_Bull": 36,
    "CDLSEPARATINGLINES_Bear": 40,
    "CDLLONGLEGGEDDOJI_Bull": 37,
    "CDLLONGLEGGEDDOJI_Bear": 37,
    "CDLHARAMI_Bull": 38,
    "CDLHARAMI_Bear": 72,
    "CDLLADDERBOTTOM_Bull": 41,
    "CDLLADDERBOTTOM_Bear": 41,
    "CDLCLOSINGMARUBOZU_Bull": 70,
    "CDLCLOSINGMARUBOZU_Bear": 43,
    "CDLTAKURI_Bull": 47,
    "CDLTAKURI_Bear": 47,
    "CDLDOJISTAR_Bull": 49,
    "CDLDOJISTAR_Bear": 51,
    "CDLHARAMICROSS_Bull": 50,
    "CDLHARAMICROSS_Bear": 80,
    "CDLADVANCEBLOCK_Bull": 54,
    "CDLADVANCEBLOCK_Bear": 54,
    "CDLSHORTLINE_Bull": 55,
    "CDLSHORTLINE_Bear": 55,
    "CDLSHOOTINGSTAR_Bull": 55,
    "CDLSHOOTINGSTAR_Bear": 55,
    "CDLMARUBOZU_Bull": 71,
    "CDLMARUBOZU_Bear": 57,
    "CDLUNIQUE3RIVER_Bull": 60,
    "CDLUNIQUE3RIVER_Bear": 60,
    "CDL2CROWS_Bull": 61,
    "CDL2CROWS_Bear": 61,
    "CDLBELTHOLD_Bull": 62,
    "CDLBELTHOLD_Bear": 63,
    "CDLHAMMER_Bull": 65,
    "CDLHAMMER_Bear": 65,
    "CDLHIGHWAVE_Bull": 67,
    "CDLHIGHWAVE_Bear": 67,
    "CDLSPINNINGTOP_Bull": 69,
    "CDLSPINNINGTOP_Bear": 73,
    "CDLUPSIDEGAP2CROWS_Bull": 74,
    "CDLUPSIDEGAP2CROWS_Bear": 74,
    "CDLGRAVESTONEDOJI_Bull": 77,
    "CDLGRAVESTONEDOJI_Bear": 77,
    "CDLHIKKAKEMOD_Bull": 82,
    "CDLHIKKAKEMOD_Bear": 81,
    "CDLHIKKAKE_Bull": 85,
    "CDLHIKKAKE_Bear": 83,
    "CDLENGULFING_Bull": 84,
    "CDLENGULFING_Bear": 91,
    "CDLMATHOLD_Bull": 86,
    "CDLMATHOLD_Bear": 86,
    "CDLHANGINGMAN_Bull": 87,
    "CDLHANGINGMAN_Bear": 87,
    "CDLRISEFALL3METHODS_Bull": 94,
    "CDLRISEFALL3METHODS_Bear": 89,
    "CDLKICKING_Bull": 96,
    "CDLKICKING_Bear": 102,
    "CDLDRAGONFLYDOJI_Bull": 98,
    "CDLDRAGONFLYDOJI_Bear": 98,
    "CDLCONCEALBABYSWALL_Bull": 101,
    "CDLCONCEALBABYSWALL_Bear": 101,
    "CDL3STARSINSOUTH_Bull": 103,
    "CDL3STARSINSOUTH_Bear": 103,
    "CDLDOJI_Bull": 104,
    "CDLDOJI_Bear": 104
}

candle_rankings_2 = {
    "CDL3LINESTRIKE_Bull": {
        "rank": 1,
        "encoding": 0
    },
    "CDL3LINESTRIKE_Bear": {
        "rank": 2,
        "encoding": 1
    },
    "CDL3BLACKCROWS_Bull": {
        "rank": 3,
        "encoding": 2
    },
    "CDL3BLACKCROWS_Bear": {
        "rank": 3,
        "encoding": 3
    },
    "CDLEVENINGSTAR_Bull": {
        "rank": 4,
        "encoding": 4
    },
    "CDLEVENINGSTAR_Bear": {
        "rank": 4,
        "encoding": 5
    },
    "CDLLONGLINE_Bull": {
        "rank": 5,
        "encoding": 6
    },
    "CDLLONGLINE_Bear": {
        "rank": 5,
        "encoding": 7
    },
    "CDLTASUKIGAP_Bull": {
        "rank": 5,
        "encoding": 8
    },
    "CDLTASUKIGAP_Bear": {
        "rank": 5,
        "encoding": 9
    },
    "CDLINVERTEDHAMMER_Bull": {
        "rank": 6,
        "encoding": 10
    },
    "CDLINVERTEDHAMMER_Bear": {
        "rank": 6,
        "encoding": 12
    },
    "CDLMATCHINGLOW_Bull": {
        "rank": 7,
        "encoding": 13
    },
    "CDLMATCHINGLOW_Bear": {
        "rank": 7,
        "encoding": 14
    },
    "CDLABANDONEDBABY_Bull": {
        "rank": 8,
        "encoding": 15
    },
    "CDLABANDONEDBABY_Bear": {
        "rank": 8,
        "encoding": 16
    },
    "CDLBREAKAWAY_Bull": {
        "rank": 10,
        "encoding": 17
    },
    "CDLBREAKAWAY_Bear": {
        "rank": 10,
        "encoding": 18
    },
    "CDLMORNINGSTAR_Bull": {
        "rank": 12,
        "encoding": 19
    },
    "CDLMORNINGSTAR_Bear": {
        "rank": 12,
        "encoding": 20
    },
    "CDLPIERCING_Bull": {
        "rank": 13,
        "encoding": 21
    },
    "CDLPIERCING_Bear": {
        "rank": 13,
        "encoding": 22
    },
    "CDLSTICKSANDWICH_Bull": {
        "rank": 14,
        "encoding": 23
    },
    "CDLSTICKSANDWICH_Bear": {
        "rank": 14,
        "encoding": 24
    },
    "CDLTHRUSTING_Bull": {
        "rank": 15,
        "encoding": 25
    },
    "CDLTHRUSTING_Bear": {
        "rank": 15,
        "encoding": 26
    },
    "CDLINNECK_Bull": {
        "rank": 17,
        "encoding": 27
    },
    "CDLINNECK_Bear": {
        "rank": 17,
        "encoding": 28
    },
    "CDL3INSIDE_Bull": {
        "rank": 20,
        "encoding": 29
    },
    "CDL3INSIDE_Bear": {
        "rank": 56,
        "encoding": 30
    },
    "CDLHOMINGPIGEON_Bull": {
        "rank": 21,
        "encoding": 31
    },
    "CDLHOMINGPIGEON_Bear": {
        "rank": 21,
        "encoding": 32
    },
    "CDLDARKCLOUDCOVER_Bull": {
        "rank": 22,
        "encoding": 33
    },
    "CDLDARKCLOUDCOVER_Bear": {
        "rank": 22,
        "encoding": 34
    },
    "CDLIDENTICAL3CROWS_Bull": {
        "rank": 24,
        "encoding": 35
    },
    "CDLIDENTICAL3CROWS_Bear": {
        "rank": 24,
        "encoding": 36
    },
    "CDLMORNINGDOJISTAR_Bull": {
        "rank": 25,
        "encoding": 37
    },
    "CDLMORNINGDOJISTAR_Bear": {
        "rank": 25,
        "encoding": 38
    },
    "CDLXSIDEGAP3METHODS_Bull": {
        "rank": 27,
        "encoding": 39
    },
    "CDLXSIDEGAP3METHODS_Bear": {
        "rank": 26,
        "encoding": 40
    },
    "CDLTRISTAR_Bull": {
        "rank": 28,
        "encoding": 41
    },
    "CDLTRISTAR_Bear": {
        "rank": 76,
        "encoding": 42
    },
    "CDLGAPSIDESIDEWHITE_Bull": {
        "rank": 46,
        "encoding": 43
    },
    "CDLGAPSIDESIDEWHITE_Bear": {
        "rank": 29,
        "encoding": 44
    },
    "CDLEVENINGDOJISTAR_Bull": {
        "rank": 30,
        "encoding": 45
    },
    "CDLEVENINGDOJISTAR_Bear": {
        "rank": 30,
        "encoding": 46
    },
    "CDL3WHITESOLDIERS_Bull": {
        "rank": 32,
        "encoding": 47
    },
    "CDL3WHITESOLDIERS_Bear": {
        "rank": 32,
        "encoding": 48
    },
    "CDLONNECK_Bull": {
        "rank": 33,
        "encoding": 49
    },
    "CDLONNECK_Bear": {
        "rank": 33,
        "encoding": 50
    },
    "CDL3OUTSIDE_Bull": {
        "rank": 34,
        "encoding": 51
    },
    "CDL3OUTSIDE_Bear": {
        "rank": 39,
        "encoding": 52
    },
    "CDLRICKSHAWMAN_Bull": {
        "rank": 35,
        "encoding": 53
    },
    "CDLRICKSHAWMAN_Bear": {
        "rank": 35,
        "encoding": 54
    },
    "CDLSEPARATINGLINES_Bull": {
        "rank": 36,
        "encoding": 55
    },
    "CDLSEPARATINGLINES_Bear": {
        "rank": 40,
        "encoding": 56
    },
    "CDLLONGLEGGEDDOJI_Bull": {
        "rank": 37,
        "encoding": 57
    },
    "CDLLONGLEGGEDDOJI_Bear": {
        "rank": 37,
        "encoding": 58
    },
    "CDLHARAMI_Bull": {
        "rank": 38,
        "encoding": 59
    },
    "CDLHARAMI_Bear": {
        "rank": 72,
        "encoding": 60
    },
    "CDLLADDERBOTTOM_Bull": {
        "rank": 41,
        "encoding": 61
    },
    "CDLLADDERBOTTOM_Bear": {
        "rank": 41,
        "encoding": 62
    },
    "CDLCLOSINGMARUBOZU_Bull": {
        "rank": 70,
        "encoding": 63
    },
    "CDLCLOSINGMARUBOZU_Bear": {
        "rank": 43,
        "encoding": 64
    },
    "CDLTAKURI_Bull": {
        "rank": 47,
        "encoding": 65
    },
    "CDLTAKURI_Bear": {
        "rank": 47,
        "encoding": 66
    },
    "CDLDOJISTAR_Bull": {
        "rank": 49,
        "encoding": 67
    },
    "CDLDOJISTAR_Bear": {
        "rank": 51,
        "encoding": 68
    },
    "CDLHARAMICROSS_Bull": {
        "rank": 50,
        "encoding": 69
    },
    "CDLHARAMICROSS_Bear": {
        "rank": 80,
        "encoding": 70
    },
    "CDLADVANCEBLOCK_Bull": {
        "rank": 54,
        "encoding": 71
    },
    "CDLADVANCEBLOCK_Bear": {
        "rank": 54,
        "encoding": 72
    },
    "CDLSHORTLINE_Bull": {
        "rank": 55,
        "encoding": 73
    },
    "CDLSHORTLINE_Bear": {
        "rank": 55,
        "encoding": 74
    },
    "CDLSHOOTINGSTAR_Bull": {
        "rank": 55,
        "encoding": 75
    },
    "CDLSHOOTINGSTAR_Bear": {
        "rank": 55,
        "encoding": 76
    },
    "CDLMARUBOZU_Bull": {
        "rank": 71,
        "encoding": 77
    },
    "CDLMARUBOZU_Bear": {
        "rank": 57,
        "encoding": 78
    },
    "CDLUNIQUE3RIVER_Bull": {
        "rank": 60,
        "encoding": 79
    },
    "CDLUNIQUE3RIVER_Bear": {
        "rank": 60,
        "encoding": 80
    },
    "CDL2CROWS_Bull": {
        "rank": 61,
        "encoding": 81
    },
    "CDL2CROWS_Bear": {
        "rank": 61,
        "encoding": 82
    },
    "CDLBELTHOLD_Bull": {
        "rank": 62,
        "encoding": 83
    },
    "CDLBELTHOLD_Bear": {
        "rank": 63,
        "encoding": 84
    },
    "CDLHAMMER_Bull": {
        "rank": 65,
        "encoding": 85
    },
    "CDLHAMMER_Bear": {
        "rank": 65,
        "encoding": 86
    },
    "CDLHIGHWAVE_Bull": {
        "rank": 67,
        "encoding": 87
    },
    "CDLHIGHWAVE_Bear": {
        "rank": 67,
        "encoding": 88
    },
    "CDLSPINNINGTOP_Bull": {
        "rank": 69,
        "encoding": 89
    },
    "CDLSPINNINGTOP_Bear": {
        "rank": 73,
        "encoding": 90
    },
    "CDLUPSIDEGAP2CROWS_Bull": {
        "rank": 74,
        "encoding": 91
    },
    "CDLUPSIDEGAP2CROWS_Bear": {
        "rank": 74,
        "encoding": 92
    },
    "CDLGRAVESTONEDOJI_Bull": {
        "rank": 77,
        "encoding": 93
    },
    "CDLGRAVESTONEDOJI_Bear": {
        "rank": 77,
        "encoding": 94
    },
    "CDLHIKKAKEMOD_Bull": {
        "rank": 82,
        "encoding": 95
    },
    "CDLHIKKAKEMOD_Bear": {
        "rank": 81,
        "encoding": 96
    },
    "CDLHIKKAKE_Bull": {
        "rank": 85,
        "encoding": 97
    },
    "CDLHIKKAKE_Bear": {
        "rank": 83,
        "encoding": 98
    },
    "CDLENGULFING_Bull": {
        "rank": 84,
        "encoding": 99
    },
    "CDLENGULFING_Bear": {
        "rank": 91,
        "encoding": 100
    },
    "CDLMATHOLD_Bull": {
        "rank": 86,
        "encoding": 101
    },
    "CDLMATHOLD_Bear": {
        "rank": 86,
        "encoding": 102
    },
    "CDLHANGINGMAN_Bull": {
        "rank": 87,
        "encoding": 103
    },
    "CDLHANGINGMAN_Bear": {
        "rank": 87,
        "encoding": 104
    },
    "CDLRISEFALL3METHODS_Bull": {
        "rank": 94,
        "encoding": 105
    },
    "CDLRISEFALL3METHODS_Bear": {
        "rank": 89,
        "encoding": 106
    },
    "CDLKICKING_Bull": {
        "rank": 96,
        "encoding": 107
    },
    "CDLKICKING_Bear": {
        "rank": 102,
        "encoding": 108
    },
    "CDLDRAGONFLYDOJI_Bull": {
        "rank": 98,
        "encoding": 109
    },
    "CDLDRAGONFLYDOJI_Bear": {
        "rank": 98,
        "encoding": 110
    },
    "CDLCONCEALBABYSWALL_Bull": {
        "rank": 101,
        "encoding": 111
    },
    "CDLCONCEALBABYSWALL_Bear": {
        "rank": 101,
        "encoding": 112
    },
    "CDL3STARSINSOUTH_Bull": {
        "rank": 103,
        "encoding": 113
    },
    "CDL3STARSINSOUTH_Bear": {
        "rank": 103,
        "encoding": 114
    },
    "CDLDOJI_Bull": {
        "rank": 104,
        "encoding": 115
    },
    "CDLDOJI_Bear": {
        "rank": 104,
        "encoding": 116
    },

    # Should be last
    "NO_PATTERN": {
        "rank": sys.maxsize,
        "encoding": 117
    },
}
