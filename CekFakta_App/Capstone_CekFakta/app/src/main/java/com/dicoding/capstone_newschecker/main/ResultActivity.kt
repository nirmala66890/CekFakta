package com.dicoding.capstone_newschecker.main

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.dicoding.capstone_newschecker.databinding.ActivityResultBinding

class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val predictionResult = intent.getIntExtra("predictionResult", -1)

        binding.predictionResultText.text = when (predictionResult) {
            1 -> "Berita ini diklasifikasikan sebagai BENAR"
            0 -> "Berita ini diklasifikasikan sebagai HOAX"
            else -> "Tidak dapat memproses hasil"
        }

        binding.checkAnotherButton.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
}
