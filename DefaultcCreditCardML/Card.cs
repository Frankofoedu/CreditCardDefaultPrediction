using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DefaultcCreditCardML
{

    public class CreditCardData
    {
        [LoadColumn(1)]
        public float LIMIT_BAL;
        [LoadColumn(2)]
        public float SEX;
        [LoadColumn(3)]
        public float EDUCATION;
        [LoadColumn(4)]
        public float MARRIAGE;

        [LoadColumn(5)]
        public float AGE;

        [LoadColumn(6)]
        public float PAY_1;

        [LoadColumn(7)]
        public float PAY_2;

        [LoadColumn(8)]
        public float PAY_3;

        [LoadColumn(9)]
        public float PAY_4;

        [LoadColumn(10)]
        public float PAY_5;

        [LoadColumn(11)]
        public float PAY_6;

        [LoadColumn(12)]
        public float BILL_AMT1;


        [LoadColumn(13)]
        public float BILL_AMT2;

        [LoadColumn(14)]
        public float BILL_AMT3;

        [LoadColumn(15)]
        public float BILL_AMT4;
        [LoadColumn(16)]
        public float BILL_AMT5;
        [LoadColumn(17)]
        public float BILL_AMT6;

        //ID,LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,
        //PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,
        //BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,
        //PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,default payment next month

        [LoadColumn(18)]
        public float PAY_AMT1;

        [LoadColumn(19)]
        public float PAY_AMT2;
        [LoadColumn(20)]
        public float PAY_AMT3;
        [LoadColumn(21)]
        public float PAY_AMT4;
        [LoadColumn(22)]
        public float PAY_AMT5;

        [LoadColumn(23)]
        public float PAY_AMT6;
        [LoadColumn(24), ColumnName("Label")]
        public bool isDefault;
    }

    public class CreditCardPrediction : CreditCardData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        //public float Probability { get; set; }

        //public float Score { get; set; }
    }
}
