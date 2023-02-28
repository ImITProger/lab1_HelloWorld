package com.imitproger.newbottonhelloworld

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import com.imitproger.newbottonhelloworld.databinding.ActivityMainBinding



class MainActivity : AppCompatActivity() {

    lateinit var binding: ActivityMainBinding


    override fun onCreate(savedInstanceState: Bundle?) {


        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val Button = binding.button

        Button.setOnClickListener() {
            val text = binding.edit1.text
            binding.textView1.text = text
         }

    }
}
