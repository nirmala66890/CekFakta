package com.dicoding.capstone_newschecker.main

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.dicoding.capstone_newschecker.databinding.ActivityMainBinding
import com.dicoding.capstone_newschecker.remote.ApiConfig
import kotlinx.coroutines.launch
import retrofit2.HttpException

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.checkButton.setOnClickListener { checkNews() }
    }

    private fun checkNews() {
        val newsTitle = binding.newsTitleEditText.text.toString().trim()
        if (newsTitle.isEmpty()) {
            Toast.makeText(this, "Masukkan judul berita!", Toast.LENGTH_SHORT).show()
            return
        }

        binding.progressBar.visibility = View.VISIBLE

        lifecycleScope.launch {
            try {
                val apiService = ApiConfig.getApiService()
                val response = apiService.predictNews(newsTitle)

                val intent = Intent(this@MainActivity, ResultActivity::class.java)
                intent.putExtra("predictionResult", response.prediction)
                startActivity(intent)
            } catch (e: HttpException) {
                Toast.makeText(this@MainActivity, "Terjadi kesalahan: ${e.message}", Toast.LENGTH_SHORT).show()
            } finally {
                binding.progressBar.visibility = View.GONE
            }
        }
    }
}
