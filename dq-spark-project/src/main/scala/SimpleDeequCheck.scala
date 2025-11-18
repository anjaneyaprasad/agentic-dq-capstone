import org.apache.spark.sql.{SparkSession, DataFrame}
import com.amazon.deequ.VerificationSuite
import com.amazon.deequ.checks.{Check, CheckLevel}

object SimpleDeequCheck {

  def main(args: Array[String]): Unit = {

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("SimpleDeequCheck")
      .master("local[*]")  // use all local cores
      .getOrCreate()

    import spark.implicits._

    // Tiny sample data
    val df: DataFrame = Seq(
      (1, "Alice", 25),
      (2, "Bob",   30),
      (3, null.asInstanceOf[String], 40)
    ).toDF("id", "name", "age")

    df.show(false)

    // Define some basic data quality checks
    val check = Check(CheckLevel.Error, "Basic Data Quality Checks")
      .isComplete("id")          // no nulls in id
      .isComplete("name")        // no nulls in name
      .isNonNegative("age")      // age >= 0

    val verificationResult = VerificationSuite()
      .onData(df)
      .addCheck(check)
      .run()

    println("==== Deequ Result ====")
    verificationResult.checkResults.foreach { case (check, result) =>
      println(s"Check: ${check.description}")
      println(s"Status: ${result.status}")
      result.constraintResults.foreach { c =>
        println(s"  - ${c.constraint}: ${c.status}")
      }
    }

    spark.stop()
  }
}
